const std = @import("std");

const Idx = @import("idx.zig").Idx;
const Matrix = @import("matrix.zig").Matrix;
const Program = @import("program.zig").Program;

const Instruction = Program.Instruction;
const Operation = Instruction.Operation;
const Symbol = Instruction.Symbol;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var prng = std.rand.DefaultPrng.init(0);
    const random = prng.random();

    const Layer = Perceptron.Layer;

    const perceptron = try Perceptron.initializeForTraining(allocator, &[_]Layer{
        try Layer.initialize(allocator, 28 * 28, 16, .leaky_relu, random),
        try Layer.initialize(allocator, 16, 16, .leaky_relu, random),
        try Layer.initialize(allocator, 16, 10, .softmax, random),
    }, .mean_squared_error);

    const training_images_file = try std.fs.cwd().openFile("train-images-idx3-ubyte", .{});
    const training_labels_file = try std.fs.cwd().openFile("train-labels-idx1-ubyte", .{});
    defer {
        training_images_file.close();
        training_labels_file.close();
    }

    const training_images = try Idx.initialize(allocator, training_images_file);
    const training_labels = try Idx.initialize(allocator, training_labels_file);

    const testing_images_file = try std.fs.cwd().openFile("t10k-images-idx3-ubyte", .{});
    const testing_labels_file = try std.fs.cwd().openFile("t10k-labels-idx1-ubyte", .{});
    defer {
        testing_images_file.close();
        testing_labels_file.close();
    }

    const testing_images = try Idx.initialize(allocator, testing_images_file);
    const testing_labels = try Idx.initialize(allocator, testing_labels_file);

    const epochs = 10;
    const mini_batch_size = 100;

    const learning_rate = 0.4; // TODO implement lr scheduling

    @memset(perceptron.loss.gradient.elements, 1);

    for (0..epochs) |epoch| {
        for (0..60_000 / mini_batch_size) |mini_batch| {
            for (perceptron.layers) |layer| {
                inline for (.{ layer.accumulated_w_gradient, layer.accumulated_b_gradient }) |accumulated_gradient|
                    @memset(accumulated_gradient.elements, 0);
            }

            for (0..mini_batch_size) |mini_batch_example| {
                for (perceptron.layers[0].n.value.elements, training_images.data[(mini_batch * mini_batch_size + mini_batch_example) * 28 * 28 ..][0 .. 28 * 28]) |*element, training_image_element|
                    element.* = @as(f32, @floatFromInt(training_image_element)) / 255;

                perceptron.program.execute();

                @memset(perceptron.y.value.elements, 0);
                perceptron.y.value.elements[training_labels.data[mini_batch * mini_batch_size + mini_batch_example]] = 1;

                perceptron.program.computeGradients();

                for (perceptron.layers) |layer| {
                    inline for (.{ .{ layer.accumulated_w_gradient, layer.w }, .{ layer.accumulated_b_gradient, layer.b } }) |parameter_tuple| {
                        for (parameter_tuple[0].elements, parameter_tuple[1].gradient.elements) |*accumulated_gradient_element, gradient_element|
                            accumulated_gradient_element.* += gradient_element;
                    }
                }
            }

            for (perceptron.layers) |layer| {
                inline for (.{ .{ layer.w, layer.accumulated_w_gradient }, .{ layer.b, layer.accumulated_b_gradient } }) |parameter_tuple| {
                    for (parameter_tuple[0].value.elements, parameter_tuple[1].elements) |*value_element, accumulated_gradient_element|
                        value_element.* -= learning_rate * accumulated_gradient_element / mini_batch_size;
                }
            }
        }

        const test_size = 1000;
        var correct: u32 = 0;

        for (0..test_size) |i| {
            for (perceptron.layers[0].n.value.elements, testing_images.data[i * 28 * 28 ..][0 .. 28 * 28]) |*element, testing_image_element|
                element.* = @as(f32, @floatFromInt(testing_image_element)) / 255;

            perceptron.program.execute();

            if (perceptron.output.value.argmax() == testing_labels.data[i])
                correct += 1;
        }

        std.debug.print("[{}] test accuracy: {}\n", .{ epoch, @as(f32, @floatFromInt(correct)) / test_size });
    }
}

const Perceptron = struct {
    program: Program,

    layers: []const Layer,
    output: Symbol,

    y: Symbol,
    loss: Symbol,

    loss_function: Operation,

    pub fn initializeForTraining(allocator: std.mem.Allocator, layers: []const Layer, loss_function: Operation) !Perceptron { // tagged union in place of layers slice to allow for loading of pretrianed pretrained parameters for further fine tuning
        var perceptron = Perceptron{
            .program = .{ .instructions = try allocator.alloc(Instruction, 2 * layers.len + 1) },
            .layers = layers,

            .output = try Symbol.initialize(allocator, layers[layers.len - 1].z.value.row_count, 1),
            .y = try Symbol.initialize(allocator, layers[layers.len - 1].z.value.row_count, 1),

            .loss = try Symbol.initialize(allocator, 1, 1),

            .loss_function = loss_function,
        };
        for (0..layers.len) |i| {
            perceptron.program.instructions[2 * i] = .{
                .operation = .affine_transformation,
                .inputs = try allocator.alloc(Symbol, 3),
                .output = layers[i].z,
            };
            @memcpy(perceptron.program.instructions[2 * i].inputs, &[_]Symbol{ layers[i].w, layers[i].n, layers[i].b });

            const output = if (i == layers.len - 1) perceptron.output else layers[i + 1].n;
            perceptron.program.instructions[2 * i + 1] = .{
                .operation = layers[i].activation_function,
                .inputs = try allocator.alloc(Symbol, 1),
                .output = output,
            };
            perceptron.program.instructions[2 * i + 1].inputs[0] = layers[i].z;
        }

        perceptron.program.instructions[perceptron.program.instructions.len - 1] = .{
            .operation = loss_function,
            .inputs = try allocator.alloc(Symbol, 2),
            .output = perceptron.loss,
        };
        @memcpy(perceptron.program.instructions[perceptron.program.instructions.len - 1].inputs, &[_]Symbol{ perceptron.y, perceptron.output });

        return perceptron;
    }

    //pub fn initializeForInference() !Perceptron {}

    //pub fn serialize(self: Perceptron, file: std.fs.File) void {
    //}

    //fn serializeStruct(structure: anytype, writer: std.io.Writer) void {
    //}

    pub const Layer = struct {
        n: Symbol,
        w: Symbol,
        b: Symbol,
        z: Symbol,

        activation_function: Operation,

        accumulated_w_gradient: Matrix,
        accumulated_b_gradient: Matrix,

        pub fn initialize(allocator: std.mem.Allocator, current_layer_neuron_count: usize, next_layer_neuron_count: usize, activation_function: Operation, random: std.rand.Random) !Layer {
            const layer = Layer{
                .n = try Symbol.initialize(allocator, current_layer_neuron_count, 1),
                .w = try Symbol.initialize(allocator, next_layer_neuron_count, current_layer_neuron_count),
                .b = try Symbol.initialize(allocator, next_layer_neuron_count, 1),
                .z = try Symbol.initialize(allocator, next_layer_neuron_count, 1),

                .activation_function = activation_function,

                .accumulated_w_gradient = try Matrix.initialize(allocator, next_layer_neuron_count, current_layer_neuron_count),
                .accumulated_b_gradient = try Matrix.initialize(allocator, next_layer_neuron_count, 1),
            };

            for (layer.w.value.elements) |*element| // switch activation_function and then decide initialization scheme?
                element.* = 3 * (random.float(f32) - 1) * @sqrt(2 / @as(f32, @floatFromInt(layer.w.value.row_count + layer.w.value.column_count)));

            @memset(layer.b.value.elements, 0);

            return layer;
        }

        pub fn free(self: Layer, allocator: std.mem.Allocator) void {
            self.n.free(allocator);
            self.w.free(allocator);
            self.b.free(allocator);

            self.accumulated_w_gradient.free(allocator);
            self.accumulated_b_gradient.free(allocator);
        }
    };
};
