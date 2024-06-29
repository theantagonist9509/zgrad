const std = @import("std");

const Matrix = @import("matrix.zig");
const Symbol = @import("symbol.zig");

pub fn Addition(argument_count: comptime_int) type {
    return struct {
        arguments: [argument_count]Symbol,
        coefficients: [argument_count]f32,

        output: Symbol,

        symbols: []Symbol,

        const Self = @This();

        pub fn operate(self: Self) void {
            @memset(self.output.value.entries, 0);

            inline for (self.arguments, self.coefficients) |argument, coefficient| {
                for (self.output.value.entries, argument.value.entries) |*output_entry, argument_entry|
                    output_entry.* += coefficient * argument_entry;
            }
        }

        pub fn backpropagate(self: Self) void {
            inline for (self.arguments, self.coefficients) |argument, coefficient| {
                for (argument.gradient.entries, self.output.gradient.entries) |*argument_entry, output_entry|
                    argument_entry.* += coefficient * output_entry;
            }
        }
    };
}

pub fn initializeAddition(allocator: std.mem.Allocator, arguments: anytype, coefficients: anytype) !Addition(arguments.len) {
    const ret = Addition(arguments.len){
        .arguments = arguments,
        .coefficients = coefficients,

        .output = try Symbol.initialize(allocator, arguments[0].value.row_count, arguments[0].value.column_count),

        .symbols = try allocator.alloc(Symbol, arguments.len + 1),
    };

    inline for (arguments, 0..) |argument, i|
        ret.symbols[i] = argument;

    ret.symbols[arguments.len] = ret.output;

    return ret;
}

pub const HadamardMultiplication = struct {};

pub const LinearTransformation = struct {
    w: Symbol,

    // For Sequence
    input: Symbol,
    output: Symbol,

    // For zeroing gradients
    symbols: []Symbol,

    // For optimizer
    parameters: []Symbol,

    pub fn initialize(Activation: type, allocator: std.mem.Allocator, random: std.Random, n1: usize, n2: usize) !LinearTransformation {
        var ret = try initializeParametersAndOutputOnly(Activation, allocator, random, n1, n2);

        ret.input = try Symbol.initialize(allocator, n1, 1);
        ret.updateInternalReferences();

        return ret;
    }

    pub fn initializeParametersAndOutputOnly(Activation: type, allocator: std.mem.Allocator, random: std.Random, n1: usize, n2: usize) !LinearTransformation {
        const ret = LinearTransformation{
            .w = try Symbol.initialize(allocator, n2, n1),

            .input = undefined,
            .output = try Symbol.initialize(allocator, n2, 1),

            .symbols = try allocator.alloc(Symbol, 3),

            .parameters = try allocator.alloc(Symbol, 1),
        };

        for (ret.w.value.entries) |*entry|
            // initialization_coefficient = k: (k * ((n1 + n2) / 2) * Var[w] = 1)

            // Glorot uniform:
            entry.* = @sqrt(24.0 / Activation.initialization_coefficient) * (random.float(f32) - 0.5) / @sqrt(@as(f32, @floatFromInt(n1 + n2)));

        return ret;
    }

    pub fn updateInternalReferences(self: LinearTransformation) void {
        @memcpy(self.parameters, &[_]Symbol{self.w});
        @memcpy(self.symbols[1 .. self.symbols.len - 1], self.parameters);
        self.symbols[0] = self.input;
        self.symbols[self.symbols.len - 1] = self.output;
    }

    pub fn operate(self: LinearTransformation) void {
        Matrix.matrixMultiply(self.output, self.w, false, self.input, false);
    }

    pub fn backpropagate(self: LinearTransformation) void {
        Matrix.matrixMultiplyAccumulate(self.w.gradient, self.output.gradient, false, self.input.value, true);
        Matrix.matrixMultiplyAccumulate(self.input.gradient, self.w.value, true, self.output.gradient, false);
    }
};

pub const AffineTransformation = struct {
    w: Symbol,
    b: Symbol,

    input: Symbol,
    output: Symbol,

    symbols: []Symbol,

    parameters: []Symbol,

    pub fn initialize(Activation: type, allocator: std.mem.Allocator, random: std.Random, n1: usize, n2: usize) !AffineTransformation {
        var ret = try initializeParametersAndOutputOnly(Activation, allocator, random, n1, n2);

        ret.input = try Symbol.initialize(allocator, n1, 1);
        ret.updateInternalReferences();

        return ret;
    }

    pub fn initializeParametersAndOutputOnly(Activation: type, allocator: std.mem.Allocator, random: std.Random, n1: usize, n2: usize) !AffineTransformation {
        const ret = AffineTransformation{
            .w = try Symbol.initialize(allocator, n2, n1),
            .b = try Symbol.initialize(allocator, n2, 1),

            .input = undefined,
            .output = try Symbol.initialize(allocator, n2, 1),

            .symbols = try allocator.alloc(Symbol, 4),

            .parameters = try allocator.alloc(Symbol, 2),
        };

        for (ret.w.value.entries) |*entry|
            entry.* = @sqrt(24.0 / Activation.initialization_coefficient) * (random.float(f32) - 0.5) / @sqrt(@as(f32, @floatFromInt(n1 + n2)));

        @memset(ret.b.value.entries, 0);

        return ret;
    }

    pub fn updateInternalReferences(self: AffineTransformation) void {
        @memcpy(self.parameters, &[_]Symbol{ self.w, self.b });
        @memcpy(self.symbols[1 .. self.symbols.len - 1], self.parameters);
        self.symbols[0] = self.input;
        self.symbols[self.symbols.len - 1] = self.output;
    }

    pub fn operate(self: AffineTransformation) void {
        Matrix.affineTransform(self.output.value, self.input.value, self.w.value, self.b.value);
    }

    pub fn backpropagate(self: AffineTransformation) void {
        Matrix.accumulate(self.b.gradient, self.output.gradient);

        Matrix.matrixMultiplyAccumulate(self.w.gradient, self.output.gradient, false, self.input.value, true);
        Matrix.matrixMultiplyAccumulate(self.input.gradient, self.w.value, true, self.output.gradient, false);
    }
};

pub const Relu = LeakyRelu(0);

pub fn LeakyRelu(slope: comptime_float) type {
    return struct {
        input: Symbol,
        output: Symbol,

        symbols: []Symbol,

        pub const initialization_coefficient = (1 + slope * slope) / 2.0;

        const Self = @This();

        pub fn initializeOutputOnly(allocator: std.mem.Allocator, n: usize) !Self {
            return .{
                .input = undefined,
                .output = try Symbol.initialize(allocator, n, 1),

                .symbols = try allocator.alloc(Symbol, 2),
            };
        }

        pub fn updateInternalReferences(self: Self) void {
            @memcpy(self.symbols, &[_]Symbol{ self.input, self.output });
        }

        pub fn operate(self: Self) void {
            Matrix.leakyRelu(slope, self.output.value, self.input.value);
        }

        pub fn backpropagate(self: Self) void {
            for (self.input.gradient.entries, self.input.value.entries, self.output.gradient.entries) |*input_gradient_entry, input_value_entry, output_gradient_entry|
                input_gradient_entry.* += @as(f32, if (input_value_entry > 0) 1 else slope) * output_gradient_entry;
        }
    };
}

pub const Tanh = struct {
    input: Symbol,
    output: Symbol,

    symbols: []Symbol,

    pub const initialization_coefficient = 1;

    pub fn initializeOutputOnly(allocator: std.mem.Allocator, n: usize) !Tanh {
        return .{
            .input = undefined,
            .output = try Symbol.initialize(allocator, n, 1),

            .symbols = try allocator.alloc(Symbol, 2),
        };
    }

    pub fn updateInternalReferences(self: Tanh) void {
        @memcpy(self.symbols, &[_]Symbol{ self.input, self.output });
    }

    pub fn operate(self: Tanh) void {
        Matrix.tanh(self.output.value, self.input.value);
    }

    pub fn backpropagate(self: Tanh) void {
        for (self.input.gradient.entries, self.output.value.entries, self.output.gradient.entries) |*input_gradient_entry, output_value_entry, output_gradient_entry| {
            const sigmoid = (1 - output_value_entry) / 2;
            input_gradient_entry.* += 4 * sigmoid * (1 - sigmoid) * output_gradient_entry;
        }
    }
};

pub const Sigmoid = struct {
    input: Symbol,
    output: Symbol,

    symbols: []Symbol,

    pub const initialization_coefficient = 0.25;

    pub fn initializeOutputOnly(allocator: std.mem.Allocator, n: usize) !Sigmoid {
        return .{
            .input = undefined,
            .output = try Symbol.initialize(allocator, n, 1),

            .symbols = try allocator.alloc(Symbol, 2),
        };
    }

    pub fn updateInternalReferences(self: Sigmoid) void {
        @memcpy(self.symbols, &[_]Symbol{ self.input, self.output });
    }

    pub fn operate(self: Sigmoid) void {
        Matrix.sigmoid(self.output.value, self.input.value);
    }

    pub fn backpropagate(self: Sigmoid) void {
        for (self.input.gradient.entries, self.output.value.entries, self.output.gradient.entries) |*input_gradient_entry, output_value_entry, output_gradient_entry|
            input_gradient_entry.* += output_value_entry * (1 - output_value_entry) * output_gradient_entry;
    }
};

pub const Softmax = struct {
    input: Symbol,
    output: Symbol,

    symbols: []Symbol,

    pub const initialization_coefficient = 0.25;

    pub fn initializeOutputOnly(allocator: std.mem.Allocator, n: usize) !Softmax {
        return .{
            .input = undefined,
            .output = try Symbol.initialize(allocator, n, 1),

            .symbols = try allocator.alloc(Symbol, 2),
        };
    }

    pub fn updateInternalReferences(self: Softmax) void {
        @memcpy(self.symbols, &[_]Symbol{ self.input, self.output });
    }

    pub fn operate(self: Softmax) void {
        Matrix.softmax(self.output.value, self.input.value);
    }

    pub fn backpropagate(self: Softmax) void {
        for (self.input.gradient.entries, self.output.value.entries, self.output.gradient.entries) |*input_gradient_entry, output_value_entry, output_gradient_entry|
            input_gradient_entry.* += output_value_entry * (1 - output_value_entry) * output_gradient_entry;
    }
};

pub const MeanSquaredError = struct {
    target: Symbol, // TODO gradient toggling

    input: Symbol,
    output: Symbol,

    symbols: []Symbol,

    pub fn initializeTargetAndOutput(allocator: std.mem.Allocator, n: usize) !MeanSquaredError {
        const ret = MeanSquaredError{
            .target = try Symbol.initialize(allocator, n, 1),

            .input = undefined,
            .output = try Symbol.initialize(allocator, 1, 1),

            .symbols = try allocator.alloc(Symbol, 3),
        };

        ret.output.gradient.entries[0] = 1;

        return ret;
    }

    pub fn updateInternalReferences(self: MeanSquaredError) void {
        @memcpy(self.symbols, &[_]Symbol{ self.input, self.target, self.output });
    }

    pub fn operate(self: MeanSquaredError) void {
        Matrix.meanSquaredError(self.output.value, self.input.value, self.target.value);
    }

    pub fn backpropagate(self: MeanSquaredError) void {
        for (self.input.gradient.entries, self.input.value.entries, self.target.gradient.entries, self.target.value.entries) |*input_gradient_entry, input_value_entry, *target_gradient_entry, target_value_entry| {
            const input_gradient_entry_update = 2 * (input_value_entry - target_value_entry) * self.output.gradient.entries[0];
            input_gradient_entry.* += input_gradient_entry_update;
            target_gradient_entry.* -= input_gradient_entry_update;
        }
    }
};

pub const CrossEntropy = struct { // TODO doesn't learn
    target: Symbol, // TODO gradient toggling

    input: Symbol,
    output: Symbol,

    symbols: []Symbol,

    pub fn initializeTargetAndOutput(allocator: std.mem.Allocator, n: usize) !CrossEntropy {
        const ret = CrossEntropy{
            .target = try Symbol.initialize(allocator, n, 1),

            .input = undefined,
            .output = try Symbol.initialize(allocator, 1, 1),

            .symbols = try allocator.alloc(Symbol, 3),
        };

        ret.output.gradient.entries[0] = 1;

        return ret;
    }

    pub fn updateInternalReferences(self: MeanSquaredError) void {
        @memcpy(self.symbols, &[_]Symbol{ self.input, self.target, self.output });
    }

    pub fn operate(self: CrossEntropy) void {
        Matrix.crossEntropy(self.output.value, self.input.value, self.target.value);
    }

    pub fn backpropagate(self: CrossEntropy) void {
        for (self.input.gradient.entries, self.input.value.entries, self.target.gradient.entries, self.target.value.entries) |*input_gradient_entry, input_value_entry, *target_gradient_entry, target_value_entry| {
            input_gradient_entry.* -= (target_value_entry / input_value_entry) * self.output.gradient.entries[0];
            target_gradient_entry.* -= @log(input_value_entry + std.math.floatEps(f32)) * self.output.gradient.entries[0];
        }
    }
};

pub fn Sequence(comptime Operations: type) type {
    return struct {
        operations: Operations,

        input: Symbol,
        output: Symbol,

        symbols: []Symbol = blk: { // Since updateInternalReferences() first frees these
            var ret: []Symbol = undefined;
            ret.len = 0;
            break :blk ret;
        },

        parameters: []Symbol = blk: {
            var ret: []Symbol = undefined;
            ret.len = 0;
            break :blk ret;
        },

        const Self = @This();

        pub fn updateInternalReferences(self: *Self, allocator: std.mem.Allocator) !void {
            self.input = self.operations[0].input;
            self.output = self.operations[self.operations.len - 1].output;

            allocator.free(self.parameters);
            allocator.free(self.symbols);

            self.parameters = try combineArrayFields(allocator, self.operations, "parameters");
            self.symbols = try combineArrayFields(allocator, self.operations, "symbols");
        }

        pub fn operate(self: Self) void {
            inline for (self.operations) |operation|
                operation.operate();
        }

        pub fn backpropagate(self: Self) void {
            inline for (self.operations, 1..) |_, i|
                self.operations[self.operations.len - i].backpropagate();
        }
    };
}

pub fn initializeSequence(allocator: std.mem.Allocator, operations: anytype) !Sequence(@TypeOf(operations)) {
    var ret = Sequence(@TypeOf(operations)){
        .operations = undefined,

        .input = undefined,
        .output = undefined,
    };

    ret.operations = operations;

    //inline for (sequence.operations[1..], 1..) |_, i| // can't slice tuples yet :(
    inline for (1..ret.operations.len) |i| {
        ret.operations[i].input = operations[i - 1].output;
        if (@typeInfo(@TypeOf(@TypeOf(operations[i]).updateInternalReferences)).Fn.params.len == 1) {
            ret.operations[i].updateInternalReferences();
        } else {
            try ret.operations[i].updateInternalReferences(allocator);
        }
    }

    try ret.updateInternalReferences(allocator);

    return ret;
}

fn combineArrayFields(allocator: std.mem.Allocator, operations: anytype, comptime field_name: []const u8) !@TypeOf(@field(operations[0], field_name)) {
    var array_list = std.ArrayListUnmanaged(@TypeOf(@field(operations[0], field_name)[0])){};

    inline for (operations) |operation| {
        inline for (std.meta.fields(@TypeOf(operation))) |field_info| { // Can't use std.meta.fieldInfo as operation may not have the field we're searching for
            if (!comptime std.mem.eql(u8, field_info.name, field_name))
                continue;

            for (@field(operation, field_name)) |item_to_be_appended| blk: {
                for (array_list.items) |existing_item| {
                    if (std.meta.eql(item_to_be_appended, existing_item))
                        break :blk;
                }

                try array_list.append(allocator, item_to_be_appended);
            }

            break;
        }
    }

    array_list.shrinkAndFree(allocator, array_list.items.len);
    return array_list.items;
}

pub fn zeroGradients(symbols: []Symbol) void {
    for (symbols) |symbol|
        @memset(symbol.gradient.entries, 0);
}

// https://arxiv.org/abs/1603.09420
pub const Mgu = struct {
    h: Symbol,

    bf: Symbol,
    bh: Symbol,

    input: Symbol,
    output: Symbol,

    linear_transformation_f_h: LinearTransformation,
    linear_transformation_f_x: LinearTransformation,
    addition_f: Addition,
    sigmoid_f: Sigmoid,

    hadamard_multiplication_h: HadamardMultiplication,
    linear_transformation_h_h: LinearTransformation,
    linear_transformation_h_x: LinearTransformation,
    addition_h: Addition,
    tanh_h: Tanh,

    hadamard_multiplication_h_tilde: HadamardMultiplication,
    addition_h_next: Addition,

    symbols: [21]Symbol,

    parameters: [6]Symbol,

    pub fn initialize(allocator: std.mem.Allocator, n_x: usize, n_h: usize) !Mgu { // initializeParametersAndOutputOnly()!!!
        var ret: Mgu = undefined;

        ret.h = try Symbol.initialize(allocator, n_h, 1);

        ret.bf = try Symbol.initialize(allocator, n_h, 1);
        ret.bh = try Symbol.initialize(allocator, n_h, 1);

        ret.input = try Symbol.initialize(allocator, n_x, 1);

        ret.linear_transformation_f_h = try LinearTransformation.initializeParametersAndOutputOnly(allocator, n_h, n_h);
        ret.linear_transformation_f_x = try LinearTransformation.initializeParametersAndOutputOnly(allocator, n_h, n_x);
        ret.addition_f = try initializeAddition(allocator, [_]Symbol{ ret.linear_transformation_f_h.output, ret.linear_transformation_f_x.output, ret.bf }, [_]f32{ 1, 1, 1 });
        ret.sigmoid_f = try Sigmoid.initializeOutputOnly(allocator, n_h);

        ret.hadamard_multiplication_h = try HadamardMultiplication.initialize(allocator, ret.sigmoid_f.output, ret.h);
        ret.linear_transformation_h_h = try LinearTransformation.initializeParametersAndOutputOnly(allocator, n_h, n_h);
        ret.linear_transformation_h_x = try LinearTransformation.initializeParametersAndOutputOnly(allocator, n_h, n_x);
        ret.addition_h = try initializeAddition(allocator, [_]Symbol{ ret.linear_transformation_h_h.output, ret.linear_transformation_h_x.output, ret.bh }, [_]f32{ 1, 1, 1 });
        ret.tanh_h = try Tanh.initializeOutputOnly(allocator, n_h);

        ret.hadamard_multiplication_h_tilde = try HadamardMultiplication.initialize(allocator, ret.sigmoid_f.output, ret.tanh_h.output);
        ret.addition_h_next = try initializeAddition(allocator, [_]Symbol{ ret.h, ret.hadamard_multiplication_h.output, ret.hadamard_multiplication_h_tilde.output }, [_]f32{ 1, -1, 1 });

        ret.linear_transformation_f_h.input = ret.h;
        ret.linear_transformation_f_x.input = ret.input;
        ret.sigmoid_f.input = ret.addition_f.output;

        ret.linear_transformation_h_h.input = ret.hadamard_multiplication_h.output;
        ret.linear_transformation_h_x.input = ret.input;
        ret.tanh_h.input = ret.addition_h.output;

        ret.output = ret.addition_h_next.output;

        inline for (.{
            ret.linear_transformation_f_h,
            ret.linear_transformation_f_x,
            ret.sigmoid_f,

            ret.linear_transformation_h_h,
            ret.linear_transformation_h_x,
            ret.tanh_h,
        }) |operation|
            operation.updateInternalReferences();

        return ret;
    }

    pub fn updateInternalReferences(self: Mgu) void {
        self.symbols = [_]Symbol{
            self.h,

            self.bf,
            self.bh,

            self.input,

            self.linear_transformation_f_h.w,
            self.linear_transformation_f_h.output,
            self.linear_transformation_f_x.w,
            self.linear_transformation_f_x.output,
            self.bf,
            self.addition_f.output,
            self.sigmoid_f.output,

            self.hadamard_multiplication_h.output,
            self.linear_transformation_h_h.w,
            self.linear_transformation_h_h.output,
            self.linear_transformation_h_x.w,
            self.linear_transformation_h_x.output,
            self.bh,
            self.addition_h.output,
            self.tanh_h.output,

            self.hadamard_multiplication_h_tilde.output,
            self.addition_h_next.output,
        };

        self.parameters = []Symbol{
            self.linear_transformation_f_h.w,
            self.linear_transformation_f_x.w,
            self.bf,

            self.linear_transformation_h_h.w,
            self.linear_transformation_h_x.w,
            self.bh,
        };
    }

    pub fn operate(self: Mgu) void {
        self.linear_transformation_f_h.operate();
        self.linear_transformation_f_x.operate();
        self.addition_f.operate();
        self.sigmoid_f.operate();

        self.hadamard_multiplication_h.operate();
        self.linear_transformation_h_h.operate();
        self.linear_transformation_h_x.operate();
        self.addition_h.operate();
        self.tanh_h.operate();

        self.hadamard_multiplication_h_tilde.operate();
        self.addition_h_next.operate();
    }

    pub fn backpropagate(self: Mgu) void {
        self.addition_h_next.backpropagate();
        self.hadamard_multiplication_h_tilde.backpropagate();

        self.tanh_h.backpropagate();
        self.addition_h.backpropagate();
        self.linear_transformation_h_x.backpropagate();
        self.linear_transformation_h_h.backpropagate();
        self.hadamard_multiplication_h.backpropagate();

        self.sigmoid_f.backpropagate();
        self.addition_f.backpropagate();
        self.linear_transformation_f_x.backpropagate();
        self.linear_transformation_f_h.backpropagate();
    }
};
