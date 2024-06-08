const std = @import("std");

const Matrix = @import("matrix.zig").Matrix;
const Program = @import("program.zig").Program;
const utilities = @import("utilities.zig");

const Instruction = Program.Instruction;
const Operation = Instruction.Operation;
const Symbol = Instruction.Symbol;

pub const Mlp = struct {
    program: Program,

    layers: []Layer, // Input to penultimate
    output: Symbol,

    was_output_allocated: bool,

    pub fn initialize(allocator: std.mem.Allocator, layers: []Layer, nullable_output: ?Symbol) !Mlp {
        var mlp = Mlp{
            .program = .{ .instructions = try allocator.alloc(Instruction, 2 * layers.len) },

            .layers = layers,
            .output = undefined,

            .was_output_allocated = undefined,
        };
        if (nullable_output) |output| {
            mlp.output = output;
            mlp.was_output_allocated = false;
        } else {
            mlp.output = try Symbol.initialize(allocator, layers[layers.len - 1].z.value.row_count, 1);
            mlp.was_output_allocated = true;
        }

        for (0..layers.len) |i| {
            mlp.program.instructions[2 * i] = .{
                .operation = .affine_transformation,
                .inputs = try allocator.alloc(Symbol, 3),
                .output = layers[i].z,
            };
            @memcpy(mlp.program.instructions[2 * i].inputs, &[_]Symbol{ layers[i].w, layers[i].n, layers[i].b });
        }
        for (0..layers.len - 1) |i| {
            mlp.program.instructions[2 * i + 1] = .{
                .operation = layers[i].activation_function,
                .inputs = try allocator.alloc(Symbol, 1),
                .output = layers[i + 1].n,
            };
            mlp.program.instructions[2 * i + 1].inputs[0] = layers[i].z;
        }
        mlp.program.instructions[2 * layers.len - 1] = .{
            .operation = layers[layers.len - 1].activation_function,
            .inputs = try allocator.alloc(Symbol, 1),
            .output = mlp.output,
        };
        mlp.program.instructions[2 * layers.len - 1].inputs[0] = layers[layers.len - 1].z;

        return mlp;
    }

    pub fn free(self: Mlp, allocator: std.mem.Allocator) void {
        allocator.free(self.program.instructions);

        if (self.was_output_allocated)
            self.output.free(allocator);
    }

    pub const Layer = struct {
        n: Symbol,
        w: Symbol,
        b: Symbol,
        z: Symbol,

        activation_function: Operation,

        accumulated_w_gradient: Matrix,
        accumulated_b_gradient: Matrix,

        was_n_allocated: bool,

        pub fn initialize(allocator: std.mem.Allocator, nullable_n: ?Symbol, current_layer_neuron_count: usize, next_layer_neuron_count: usize, activation_function: Operation, random: std.rand.Random) !Layer {
            var layer = Layer{
                .n = undefined,
                .w = try Symbol.initialize(allocator, next_layer_neuron_count, current_layer_neuron_count),
                .b = try Symbol.initialize(allocator, next_layer_neuron_count, 1),
                .z = try Symbol.initialize(allocator, next_layer_neuron_count, 1),

                .activation_function = activation_function,

                .accumulated_w_gradient = try Matrix.initialize(allocator, next_layer_neuron_count, current_layer_neuron_count),
                .accumulated_b_gradient = try Matrix.initialize(allocator, next_layer_neuron_count, 1),

                .was_n_allocated = undefined,
            };
            if (nullable_n) |n| {
                layer.n = n;
                layer.was_n_allocated = false;
            } else {
                layer.n = try Symbol.initialize(allocator, current_layer_neuron_count, 1);
                layer.was_n_allocated = true;
            }

            switch (activation_function) {
                .relu, .leaky_relu => { // TODO same for relu and lrelu?
                    for (layer.w.value.elements) |*element|
                        element.* = 4 * (random.float(f32) - 1) * @sqrt(3 / @as(f32, @floatFromInt(layer.w.value.row_count + layer.w.value.column_count))); // TODO recheck; why does 3*root(2) give better (?) convergence?
                },
                .sigmoid, .softmax => {
                    for (layer.w.value.elements) |*element|
                        element.* = 8 * (random.float(f32) - 1) * @sqrt(6 / @as(f32, @floatFromInt(layer.w.value.row_count + layer.w.value.column_count)));
                },

                else => @panic("Activation function initialization scheme not implemented"),
            }

            @memset(layer.b.value.elements, 0);

            return layer;
        }

        pub fn free(self: Layer, allocator: std.mem.Allocator) void {
            if (self.was_n_allocated)
                self.n.free(allocator);

            self.w.free(allocator);
            self.b.free(allocator);
            self.z.free(allocator);

            self.accumulated_w_gradient.free(allocator);
            self.accumulated_b_gradient.free(allocator);
        }

        pub fn serializeLayers(allocator: std.mem.Allocator, layers: []Layer, file: std.fs.File) !void {
            // Flagging [See utilities.zig]

            for (layers) |*layer| {
                layer.n.toggleSerializationFlag();
                layer.z.toggleSerializationFlag();

                layer.w.gradient.toggleSerializationFlag();
                layer.b.gradient.toggleSerializationFlag();

                layer.accumulated_w_gradient.toggleSerializationFlag();
                layer.accumulated_b_gradient.toggleSerializationFlag();
            }

            // Serialization

            var pointer_list = std.ArrayList(usize).init(allocator);
            defer pointer_list.deinit();

            try utilities.serializeObject(layers, file, &pointer_list);

            // Unflagging

            for (layers) |*layer| {
                layer.n.toggleSerializationFlag();
                layer.z.toggleSerializationFlag();

                layer.w.gradient.toggleSerializationFlag();
                layer.b.gradient.toggleSerializationFlag();

                layer.accumulated_w_gradient.toggleSerializationFlag();
                layer.accumulated_b_gradient.toggleSerializationFlag();
            }
        }

        pub fn deserializeLayers(allocator: std.mem.Allocator, file: std.fs.File) ![]Layer {
            var pointer_list = std.ArrayListUnmanaged(usize){};
            defer pointer_list.deinit(allocator);

            return utilities.deserializeObject([]Layer, allocator, file, &pointer_list);
        }
    };
};
