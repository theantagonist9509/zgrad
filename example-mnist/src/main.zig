const std = @import("std");
const zgrad = @import("zgrad");

const Idx = @import("idx.zig").Idx;

const Matrix = zgrad.core.Matrix;
const Program = zgrad.core.Program;
const Mlp = zgrad.templates.Mlp;

const Instruction = Program.Instruction;
const Operation = Instruction.Operation;
const Symbol = Instruction.Symbol;
const Layer = Mlp.Layer;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    //var prng = std.rand.DefaultPrng.init(0);
    //const random = prng.random();

    //var layers = [_]Layer{
    //    try Layer.initialize(allocator, 28 * 28, 16, .leaky_relu, random),
    //    try Layer.initialize(allocator, 16, 16, .leaky_relu, random),
    //    try Layer.initialize(allocator, 16, 10, .softmax, random),
    //};

    //const mlp = try Mlp.initializeForTraining(allocator, &layers}, .mean_squared_error);

    const file = try std.fs.cwd().createFile("mnist.zmlp", .{ .read = true, .truncate = false });
    defer file.close();

    var mlp: Mlp = undefined;
    const t1 = std.time.nanoTimestamp();
    try mlp.deserialize(allocator, file);
    const t2 = std.time.nanoTimestamp();
    std.debug.print("deser_time: {} ns\n", .{t2 - t1});

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

    const epochs = 5;
    const mini_batch_size = 5;

    const learning_rate = 0.001; // TODO implement lr scheduling

    @memset(mlp.loss.gradient.elements, 1);

    for (0..epochs) |epoch| {
        for (0..60_000 / mini_batch_size) |mini_batch| {
            for (mlp.layers) |layer| {
                inline for (.{ layer.accumulated_w_gradient, layer.accumulated_b_gradient }) |accumulated_gradient|
                    @memset(accumulated_gradient.elements, 0);
            }

            for (0..mini_batch_size) |mini_batch_example| {
                for (mlp.layers[0].n.value.elements, training_images.data[(mini_batch * mini_batch_size + mini_batch_example) * 28 * 28 ..][0 .. 28 * 28]) |*element, training_image_element|
                    element.* = @as(f32, @floatFromInt(training_image_element)) / 255;

                mlp.program.execute();

                @memset(mlp.y.value.elements, 0);
                mlp.y.value.elements[training_labels.data[mini_batch * mini_batch_size + mini_batch_example]] = 1;

                mlp.program.computeGradients();

                for (mlp.layers) |layer| {
                    inline for (.{ .{ layer.accumulated_w_gradient, layer.w }, .{ layer.accumulated_b_gradient, layer.b } }) |parameter_tuple| {
                        for (parameter_tuple[0].elements, parameter_tuple[1].gradient.elements) |*accumulated_gradient_element, gradient_element|
                            accumulated_gradient_element.* += gradient_element;
                    }
                }
            }

            for (mlp.layers) |layer| {
                inline for (.{ .{ layer.w, layer.accumulated_w_gradient }, .{ layer.b, layer.accumulated_b_gradient } }) |parameter_tuple| {
                    for (parameter_tuple[0].value.elements, parameter_tuple[1].elements) |*value_element, accumulated_gradient_element|
                        value_element.* -= learning_rate * accumulated_gradient_element / mini_batch_size;
                }
            }
        }

        const test_size = 1000;
        var correct: u32 = 0;

        for (0..test_size) |i| {
            for (mlp.layers[0].n.value.elements, testing_images.data[i * 28 * 28 ..][0 .. 28 * 28]) |*element, testing_image_element|
                element.* = @as(f32, @floatFromInt(testing_image_element)) / 255;

            mlp.program.execute();

            if (mlp.output.value.argmax() == testing_labels.data[i])
                correct += 1;
        }

        std.debug.print("[{}] test accuracy: {}\n", .{ epoch, @as(f32, @floatFromInt(correct)) / test_size });
    }

    try file.seekTo(0);
    const t3 = std.time.nanoTimestamp();
    try mlp.serialize(allocator, file);
    const t4 = std.time.nanoTimestamp();
    std.debug.print("ser_time: {} ns\n", .{t4 - t3});
}
