const std = @import("std");
const zgrad = @import("zgrad");

const IdxUbyte = @import("idxubyte.zig").IdxUbyte;

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

    var layers: []Layer = undefined;

    const file_name = "autoencoder.zmlp";

    if (std.fs.cwd().openFile(file_name, .{})) |in_file| {
        defer in_file.close();

        const t1 = std.time.microTimestamp();

        layers = try Layer.deserializeLayers(allocator, in_file);

        const t2 = std.time.microTimestamp();

        std.debug.print("deserialization_time: {} us\n", .{t2 - t1});
    } else |err| switch (err) {
        error.FileNotFound => {
            var prng = std.rand.DefaultPrng.init(0);
            const random = prng.random();
            const new_layers = [_]Layer{
                try Layer.initialize(allocator, null, 28 * 28, 32, .leaky_relu, random),
                try Layer.initialize(allocator, null, 32, 16, .leaky_relu, random),
                try Layer.initialize(allocator, null, 16, 8, .leaky_relu, random),
                try Layer.initialize(allocator, null, 8, 16, .leaky_relu, random),
                try Layer.initialize(allocator, null, 16, 32, .leaky_relu, random),
                try Layer.initialize(allocator, null, 32, 28 * 28, .sigmoid, random),
            };
            layers = try allocator.alloc(Layer, new_layers.len);
            @memcpy(layers, &new_layers);
        },
        else => |remaining_error| return remaining_error,
    }

    var autoencoder = try Mlp.initialize(allocator, layers, null);

    const training_images_file = try std.fs.cwd().openFile("train-images-idx3-ubyte", .{});
    defer training_images_file.close();

    const training_images = try IdxUbyte.initialize(allocator, training_images_file);

    const testing_images_file = try std.fs.cwd().openFile("t10k-images-idx3-ubyte", .{});
    defer testing_images_file.close();

    const testing_images = try IdxUbyte.initialize(allocator, testing_images_file);

    const epochs = 10;
    const mini_batch_size = 10;

    const learning_rate = 2; // TODO implement lr scheduling
    const accumulated_gradient_decay_coefficient = 0.75; // rename?

    const y = try Symbol.initialize(allocator, 28 * 28, 1);
    const loss = try Symbol.initialize(allocator, 1, 1);

    const loss_instruciton = Instruction{
        .operation = .mean_squared_error,
        .inputs = try allocator.alloc(Symbol, 2),
        .output = loss,
    };
    @memcpy(loss_instruciton.inputs, &[_]Symbol{ y, autoencoder.output });

    loss.gradient.elements[0] = 1;

    for (0..epochs) |epoch| {
        const training_images_count = 60_000;

        var accumulated_loss: f32 = 0;

        for (0..training_images_count / mini_batch_size) |mini_batch| {
            for (autoencoder.layers) |layer| {
                inline for (.{ layer.accumulated_w_gradient, layer.accumulated_b_gradient }) |accumulated_gradient| {
                    for (accumulated_gradient.elements) |*element|
                        element.* *= accumulated_gradient_decay_coefficient;
                }
            }

            for (0..mini_batch_size) |mini_batch_example| {
                for (autoencoder.layers[0].n.value.elements, training_images.data[(mini_batch * mini_batch_size + mini_batch_example) * 28 * 28 ..][0 .. 28 * 28]) |*element, training_image_element|
                    element.* = @as(f32, @floatFromInt(training_image_element)) / 255;

                @memcpy(y.value.elements, autoencoder.layers[0].n.value.elements);

                autoencoder.program.execute();
                loss_instruciton.execute();

                accumulated_loss += loss.value.elements[0];

                loss_instruciton.backpropagate();
                autoencoder.program.backpropagate();

                for (autoencoder.layers) |layer| {
                    inline for (.{ .{ layer.accumulated_w_gradient, layer.w }, .{ layer.accumulated_b_gradient, layer.b } }) |parameter_tuple| {
                        for (parameter_tuple[0].elements, parameter_tuple[1].gradient.elements) |*accumulated_gradient_element, gradient_element|
                            accumulated_gradient_element.* += (1 - accumulated_gradient_decay_coefficient) * gradient_element;
                    }
                }
            }

            for (autoencoder.layers) |layer| {
                inline for (.{ .{ layer.w, layer.accumulated_w_gradient }, .{ layer.b, layer.accumulated_b_gradient } }) |parameter_tuple| {
                    for (parameter_tuple[0].value.elements, parameter_tuple[1].elements) |*value_element, accumulated_gradient_element|
                        value_element.* -= learning_rate * accumulated_gradient_element / mini_batch_size;
                }
            }
        }

        const testing_images_count = 100;

        for (0..testing_images_count) |i| {
            for (autoencoder.layers[0].n.value.elements, testing_images.data[i * 28 * 28 ..][0 .. 28 * 28]) |*element, testing_image_element|
                element.* = @as(f32, @floatFromInt(testing_image_element)) / 255;

            autoencoder.program.execute();

            if (epoch == epochs - 1) {
                drawImage(autoencoder.layers[0].n.value.elements);
                drawImage(autoencoder.output.value.elements);
            }
        }

        std.debug.print("[{}] cost: {}\n", .{ epoch, accumulated_loss / training_images_count });
    }

    const out_file = try std.fs.cwd().createFile(file_name, .{});
    defer out_file.close();

    const t1 = std.time.microTimestamp();

    try Layer.serializeLayers(allocator, autoencoder.layers, out_file);

    const t2 = std.time.microTimestamp();

    std.debug.print("serialization_time: {} us\n", .{t2 - t1});

    // Serialize the autoencoder as an encoder-decoder pair

    const encoder_file = try std.fs.cwd().createFile("encoder.zmlp", .{});
    const decoder_file = try std.fs.cwd().createFile("decoder.zmlp", .{});
    defer {
        encoder_file.close();
        decoder_file.close();
    }

    try Layer.serializeLayers(allocator, autoencoder.layers[0 .. autoencoder.layers.len / 2], encoder_file);
    try Layer.serializeLayers(allocator, autoencoder.layers[autoencoder.layers.len / 2 ..], decoder_file);
}

fn drawImage(data: []f32) void {
    for (0..28) |i| {
        for (0..28) |j| {
            const character = getBrightnessCharacter(data[i * 28 + j]);
            std.debug.print("{c}{c}", .{ character, character });
        }

        std.debug.print("\n", .{});
    }
}

fn getBrightnessCharacter(brightness: f32) u8 {
    const characters = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "; // paulbourke.net/dataformats/asciiart

    for (characters, 1..) |_, i| {
        if (brightness <= @as(f32, @floatFromInt(i)) / characters.len)
            return characters[characters.len - i];
    }

    unreachable;
}
