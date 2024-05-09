const std = @import("std");

const Matrix = @import("matrix.zig").Matrix;

pub const Program = struct {
    instructions: []Instruction,

    pub fn execute(self: Program) void {
        for (self.instructions) |instruction|
            instruction.operate();
    }

    pub fn computeGradients(self: Program) void {
        var index: usize = self.instructions.len;
        while (index > 0) {
            index -= 1;

            const inputs = self.instructions[index].inputs;
            const output = self.instructions[index].output;

            switch (self.instructions[index].operation) {
                .addition => {
                    Matrix.copy(inputs[0].gradient, output.gradient);
                    Matrix.copy(inputs[1].gradient, output.gradient);
                },
                .matrix_multiplication => {
                    Matrix.matrixMultiply(inputs[0].gradient, output.gradient, false, inputs[1].value, true);
                    Matrix.matrixMultiply(inputs[1].gradient, inputs[0].value, true, output.gradient, false);
                },
                .affine_transformation => {
                    Matrix.copy(inputs[2].gradient, output.gradient);

                    Matrix.matrixMultiply(inputs[0].gradient, output.gradient, false, inputs[1].value, true);
                    Matrix.matrixMultiply(inputs[1].gradient, inputs[0].value, true, output.gradient, false);
                },

                .relu => {
                    for (inputs[0].gradient.elements, inputs[0].value.elements, output.gradient.elements) |*gradient_element, value_element, output_gradient_element|
                        gradient_element.* = @as(f32, if (value_element > 0) 1 else 0) * output_gradient_element;
                },
                .leaky_relu => {
                    for (inputs[0].gradient.elements, inputs[0].value.elements, output.gradient.elements) |*gradient_element, value_element, output_gradient_element|
                        gradient_element.* = @as(f32, if (value_element > 0) 1 else Matrix.leaky_relu_slope) * output_gradient_element;
                },
                .softsign => {
                    for (inputs[0].gradient.elements, inputs[0].value.elements, output.gradient.elements) |*gradient_element, value_element, output_gradient_element|
                        gradient_element.* = output_gradient_element / ((1 + @abs(value_element)) * (1 + @abs(value_element)));
                },
                .softmax => {
                    var maximum = inputs[0].value.elements[0];
                    for (inputs[0].value.elements[1..]) |element| {
                        if (element > maximum)
                            maximum = element;
                    }

                    var sum: f32 = 0;
                    for (inputs[0].value.elements) |element|
                        sum += @exp(element - maximum);

                    for (inputs[0].gradient.elements, inputs[0].value.elements, output.gradient.elements) |*gradient_element, value_element, output_gradient_element| {
                        const softmax = @exp(value_element - maximum) / sum;
                        gradient_element.* = softmax * (1 - softmax) * output_gradient_element;
                    }
                },

                .mean_squared_error => {
                    for (inputs[0].gradient.elements, inputs[0].value.elements, inputs[1].gradient.elements, inputs[1].value.elements) |*inputs_0_gradient_element, inputs_0_value_element, *inputs_1_gradient_element, inputs_1_value_element| {
                        inputs_0_gradient_element.* = (2 * (inputs_0_value_element - inputs_1_value_element) / @as(f32, @floatFromInt(inputs[0].value.elements.len))) * output.gradient.elements[0];
                        inputs_1_gradient_element.* = -inputs_0_gradient_element.*;
                    }
                },
                .cross_entropy => {
                    for (inputs[0].gradient.elements, inputs[0].value.elements, inputs[1].gradient.elements, inputs[1].value.elements) |*inputs_0_gradient_element, inputs_0_value_element, *inputs_1_gradient_element, inputs_1_value_element| {
                        inputs_0_gradient_element.* = -@log(inputs_1_value_element) * output.gradient.elements[0];
                        inputs_1_gradient_element.* = -(inputs_0_value_element / inputs_1_value_element) * output.gradient.elements[0];
                    }
                },

                //else => {
                //    @panic("Gradient not implemented");
                //},
            }
        }
    }

    pub const Instruction = struct {
        operation: Operation,
        inputs: []Symbol,
        output: Symbol,

        pub fn operate(self: Instruction) void {
            switch (self.operation) {
                .addition => Matrix.add(self.output.value, self.inputs[0].value, self.inputs[1].value),
                .matrix_multiplication => Matrix.matrixMultiply(self.output.value, self.inputs[0].value, false, self.inputs[1].value, false),
                .affine_transformation => Matrix.affineTransform(self.output.value, self.inputs[0].value, self.inputs[1].value, self.inputs[2].value),

                .relu => Matrix.relu(self.output.value, self.inputs[0].value),
                .leaky_relu => Matrix.leakyRelu(self.output.value, self.inputs[0].value),
                .softsign => Matrix.softsign(self.output.value, self.inputs[0].value),
                .softmax => Matrix.softmax(self.output.value, self.inputs[0].value),

                .mean_squared_error => Matrix.meanSquaredError(self.output.value, self.inputs[0].value, self.inputs[1].value),
                .cross_entropy => Matrix.crossEntropy(self.output.value, self.inputs[0].value, self.inputs[1].value),
            }
        }

        pub const Operation = enum { // remove unnecessary ones
            addition,
            matrix_multiplication,
            affine_transformation,

            relu,
            leaky_relu,
            softsign,
            softmax,

            mean_squared_error,
            cross_entropy,
        };

        pub const Symbol = struct { // boolean for toggling gradient evaluation?
            value: Matrix,
            gradient: Matrix,

            pub fn initialize(allocator: std.mem.Allocator, row_count: usize, column_count: usize) !Symbol {
                return .{
                    .value = try Matrix.initialize(allocator, row_count, column_count),
                    .gradient = try Matrix.initialize(allocator, row_count, column_count),
                };
            }

            pub fn free(self: Symbol, allocator: std.mem.Allocator) void {
                self.value.free(allocator);
                self.gradient.free(allocator);
            }
        };
    };
};
