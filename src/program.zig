const std = @import("std");

const Matrix = @import("matrix.zig").Matrix;

pub const Program = struct {
    instructions: []Instruction,

    pub fn execute(self: Program) void {
        for (self.instructions) |instruction|
            instruction.execute();
    }

    pub fn backpropagate(self: Program) void {
        var i = self.instructions.len;
        while (i > 0) {
            i -= 1;
            self.instructions[i].backpropagate();
        }
    }

    pub const Instruction = struct {
        operation: Operation,
        inputs: []Symbol,
        output: Symbol,

        pub fn execute(self: Instruction) void {
            switch (self.operation) {
                .addition => Matrix.add(self.output.value, self.inputs[0].value, self.inputs[1].value),
                .matrix_multiplication => Matrix.matrixMultiply(self.output.value, self.inputs[0].value, false, self.inputs[1].value, false),
                .affine_transformation => Matrix.affineTransform(self.output.value, self.inputs[0].value, self.inputs[1].value, self.inputs[2].value),

                .relu => Matrix.relu(self.output.value, self.inputs[0].value),
                .leaky_relu => Matrix.leakyRelu(self.output.value, self.inputs[0].value),
                .sigmoid => Matrix.sigmoid(self.output.value, self.inputs[0].value),
                .softmax => Matrix.softmax(self.output.value, self.inputs[0].value),

                //.layer_normalization => Matrix.layerNormalization(self.output.value, self.inputs[0].value),

                .mean_squared_error => Matrix.meanSquaredError(self.output.value, self.inputs[0].value, self.inputs[1].value),
                .cross_entropy => Matrix.crossEntropy(self.output.value, self.inputs[0].value, self.inputs[1].value),
            }
        }

        pub fn backpropagate(self: Instruction) void {
            switch (self.operation) {
                .addition => {
                    Matrix.copy(self.inputs[0].gradient, self.output.gradient);
                    Matrix.copy(self.inputs[1].gradient, self.output.gradient);
                },
                .matrix_multiplication => {
                    Matrix.matrixMultiply(self.inputs[0].gradient, self.output.gradient, false, self.inputs[1].value, true);
                    Matrix.matrixMultiply(self.inputs[1].gradient, self.inputs[0].value, true, self.output.gradient, false);
                },
                .affine_transformation => {
                    Matrix.copy(self.inputs[2].gradient, self.output.gradient);

                    Matrix.matrixMultiply(self.inputs[0].gradient, self.output.gradient, false, self.inputs[1].value, true);
                    Matrix.matrixMultiply(self.inputs[1].gradient, self.inputs[0].value, true, self.output.gradient, false);
                },

                .relu => {
                    for (self.inputs[0].gradient.elements, self.inputs[0].value.elements, self.output.gradient.elements) |*gradient_element, value_element, output_gradient_element|
                        gradient_element.* = @as(f32, if (value_element > 0) 1 else 0) * output_gradient_element;
                },
                .leaky_relu => {
                    for (self.inputs[0].gradient.elements, self.inputs[0].value.elements, self.output.gradient.elements) |*gradient_element, value_element, output_gradient_element|
                        gradient_element.* = @as(f32, if (value_element > 0) 1 else Matrix.leaky_relu_slope) * output_gradient_element;
                },
                .sigmoid, .softmax => {
                    for (self.inputs[0].gradient.elements, self.output.value.elements, self.output.gradient.elements) |*gradient_element, output_value_element, output_gradient_element|
                        gradient_element.* = output_value_element * (1 - output_value_element) * output_gradient_element;
                },

                //.layer_normalization => { // TODO make this a struct that stores sd so we don't have to recalculate it after the forward pass?
                //    var mean: f32 = 0;
                //    var sd: f32 = 0;

                //    for (self.inputs[0].value.elements) |element| {
                //        mean += element;
                //        sd += element * element;
                //    }

                //    mean /= self.inputs[0].value.elements.len;
                //    sd = @sqrt(sd / self.inputs[0].value.elements.len - mean * mean);

                //    for (self.inputs[0].gradient.elements, self.output.gradient.elements) |*gradient_element, output_gradient_element|
                //        gradient_element.* = output_gradient_element / sd; // TODO fix this!
                //},

                .mean_squared_error => { // TODO reverse operands?
                    for (self.inputs[0].gradient.elements, self.inputs[0].value.elements, self.inputs[1].gradient.elements, self.inputs[1].value.elements) |*inputs_0_gradient_element, inputs_0_value_element, *inputs_1_gradient_element, inputs_1_value_element| {
                        inputs_0_gradient_element.* = (2 * (inputs_0_value_element - inputs_1_value_element) / @as(f32, @floatFromInt(self.inputs[0].value.elements.len))) * self.output.gradient.elements[0];
                        inputs_1_gradient_element.* = -inputs_0_gradient_element.*;
                    }
                },
                .cross_entropy => { // TODO reverse operands?
                    for (self.inputs[0].gradient.elements, self.inputs[0].value.elements, self.inputs[1].gradient.elements, self.inputs[1].value.elements) |*inputs_0_gradient_element, inputs_0_value_element, *inputs_1_gradient_element, inputs_1_value_element| {
                        inputs_0_gradient_element.* = -@log(inputs_1_value_element + std.math.floatEps(f32)) * self.output.gradient.elements[0]; // Biasing for numerical stability
                        inputs_1_gradient_element.* = -(inputs_0_value_element / inputs_1_value_element) * self.output.gradient.elements[0];
                    }
                },

                //else => {
                //    @panic("Gradient propagation not implemented");
                //},
            }
        }

        pub const Operation = enum {
            addition,
            matrix_multiplication,
            affine_transformation,

            relu,
            leaky_relu,
            sigmoid,
            softmax,

            //layer_normalization,

            mean_squared_error,
            cross_entropy,
        };

        pub const Symbol = struct {
            value: Matrix,
            gradient: Matrix,

            pub fn initialize(allocator: std.mem.Allocator, row_count: usize, column_count: usize) !Symbol {
                return .{
                    .value = try Matrix.initialize(allocator, row_count, column_count),
                    .gradient = try Matrix.initialize(allocator, row_count, column_count),
                };
            }

            // Gradient won't be serialized (neither length, nor content)
            pub fn free(self: *Symbol, allocator: std.mem.Allocator) void {
                self.value.free(allocator); // Matrix.free() sets elements.len to 0
                self.gradient.free(allocator);
            }

            pub fn toggleSerializationFlag(self: *Symbol) void {
                self.value.toggleSerializationFlag();
                self.gradient.toggleSerializationFlag();
            }
        };
    };
};
