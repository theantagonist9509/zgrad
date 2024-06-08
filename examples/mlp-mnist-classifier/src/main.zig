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

    const file_name = "classifier.zmlp";

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
                try Layer.initialize(allocator, null, 28 * 28, 8, .leaky_relu, random),
                try Layer.initialize(allocator, null, 8, 8, .leaky_relu, random),
                try Layer.initialize(allocator, null, 8, 10, .softmax, random),
            };
            layers = try allocator.alloc(Layer, new_layers.len);
            @memcpy(layers, &new_layers);
        },
        else => |remaining_error| return remaining_error,
    }

    var classifier = try Mlp.initialize(allocator, layers, null);

    var training_images_normalized_pixels: []f32 = undefined;
    var testing_images_normalized_pixels: []f32 = undefined;

    const training_images_mean = try allocator.alloc(f32, 28 * 28);
    const training_images_sd = try allocator.alloc(f32, 28 * 28); // Standard deviation

    var training_labels: IdxUbyte = undefined;
    var testing_labels: IdxUbyte = undefined;

    {
        const training_images_file = try std.fs.cwd().openFile("train-images-idx3-ubyte", .{});
        const testing_images_file = try std.fs.cwd().openFile("t10k-images-idx3-ubyte", .{});
        defer {
            training_images_file.close();
            testing_images_file.close();
        }

        const training_images_idxubyte = try IdxUbyte.initialize(allocator, training_images_file);
        const testing_images_idxubyte = try IdxUbyte.initialize(allocator, testing_images_file);
        defer {
            training_images_idxubyte.free(allocator);
            testing_images_idxubyte.free(allocator);
        }

        training_images_normalized_pixels = try allocator.alloc(f32, training_images_idxubyte.data.len);
        testing_images_normalized_pixels = try allocator.alloc(f32, testing_images_idxubyte.data.len);

        for (training_images_normalized_pixels, training_images_idxubyte.data) |*pixel, pixel_byte|
            pixel.* = @as(f32, @floatFromInt(pixel_byte)) / 255;

        for (testing_images_normalized_pixels, testing_images_idxubyte.data) |*pixel, pixel_byte|
            pixel.* = @as(f32, @floatFromInt(pixel_byte)) / 255;

        const training_labels_file = try std.fs.cwd().openFile("train-labels-idx1-ubyte", .{});
        const testing_labels_file = try std.fs.cwd().openFile("t10k-labels-idx1-ubyte", .{});
        defer {
            training_labels_file.close();
            testing_labels_file.close();
        }

        training_labels = try IdxUbyte.initialize(allocator, training_labels_file);
        testing_labels = try IdxUbyte.initialize(allocator, testing_labels_file);
    }

    @memset(training_images_mean, 0);
    @memset(training_images_sd, 0);
    for (training_images_normalized_pixels, 0..) |pixel, i| {
        training_images_mean[i % (28 * 28)] += pixel;
        training_images_sd[i % (28 * 28)] += pixel * pixel;
    }

    for (training_images_mean, training_images_sd) |*pixel_mean, *pixel_sd| {
        pixel_mean.* /= 60_000;
        pixel_sd.* = @sqrt(pixel_sd.* / 60_000 - pixel_mean.* * pixel_mean.*);
    }

    drawImage(training_images_mean);
    drawImage(training_images_sd);

    const normalization_threshold = 0.1; // TODO value?
    inline for (.{ training_images_normalized_pixels, testing_images_normalized_pixels }) |images_pixels| {
        for (images_pixels, 0..) |*pixel, i| {
            const sd = training_images_sd[i % (28 * 28)];

            if (sd < normalization_threshold)
                continue;

            const mean = training_images_mean[i % (28 * 28)];

            pixel.* = (pixel.* - mean) / sd;
        }
    }

    const epochs = 10;
    const training_images_count = 60_000;
    const testing_images_count = 1_000;
    const mini_batch_size = 100;

    const learning_rate = 0.001_5; // TODO implement lr scheduling
    const accumulated_gradient_decay_coefficient = 0.5;

    const y = try Symbol.initialize(allocator, 10, 1); // Target output
    const loss = try Symbol.initialize(allocator, 1, 1);

    const loss_instruction = Instruction{
        .operation = .mean_squared_error,
        .inputs = try allocator.alloc(Symbol, 2),
        .output = loss,
    };
    @memcpy(loss_instruction.inputs, &[_]Symbol{ y, classifier.output });

    loss.gradient.elements[0] = 1;

    for (classifier.layers) |layer| {
        inline for (.{ layer.accumulated_w_gradient, layer.accumulated_b_gradient }) |accumulated_gradient|
            @memset(accumulated_gradient.elements, 0);
    }

    for (0..epochs) |epoch| {
        var accumulated_loss: f32 = 0;

        for (0..training_images_count / mini_batch_size) |mini_batch| {
            for (classifier.layers) |layer| {
                inline for (.{ layer.accumulated_w_gradient, layer.accumulated_b_gradient }) |accumulated_gradient| {
                    for (accumulated_gradient.elements) |*element|
                        element.* *= accumulated_gradient_decay_coefficient;
                }
            }

            for (0..mini_batch_size) |mini_batch_image_index| {
                @memcpy(classifier.layers[0].n.value.elements, training_images_normalized_pixels[(mini_batch * mini_batch_size + mini_batch_image_index) * 28 * 28 ..][0 .. 28 * 28]);

                @memset(y.value.elements, 0);
                y.value.elements[training_labels.data[mini_batch * mini_batch_size + mini_batch_image_index]] = 1;

                classifier.program.execute();
                loss_instruction.execute();

                accumulated_loss += loss.value.elements[0];

                loss_instruction.backpropagate();
                classifier.program.backpropagate();

                for (classifier.layers) |layer| {
                    inline for (.{ .{ layer.accumulated_w_gradient, layer.w }, .{ layer.accumulated_b_gradient, layer.b } }) |parameter_tuple| {
                        for (parameter_tuple[0].elements, parameter_tuple[1].gradient.elements) |*accumulated_gradient_element, gradient_element|
                            accumulated_gradient_element.* += (1 - accumulated_gradient_decay_coefficient) * gradient_element;
                    }
                }
            }

            for (classifier.layers) |layer| {
                inline for (.{ .{ layer.w, layer.accumulated_w_gradient }, .{ layer.b, layer.accumulated_b_gradient } }) |parameter_tuple| {
                    for (parameter_tuple[0].value.elements, parameter_tuple[1].elements) |*value_element, accumulated_gradient_element|
                        value_element.* -= learning_rate * accumulated_gradient_element / mini_batch_size;
                }
            }
        }

        var correct: u32 = 0;

        for (0..testing_images_count) |i| {
            @memcpy(classifier.layers[0].n.value.elements, testing_images_normalized_pixels[i * 28 * 28 ..][0 .. 28 * 28]);

            classifier.program.execute();

            if (epoch == epochs - 1) {
                drawNormalizedImage(classifier.layers[0].n.value.elements, training_images_mean, training_images_sd, normalization_threshold);
                std.debug.print("{}\n", .{classifier.output.value.argmax()});
            }

            if (classifier.output.value.argmax() == testing_labels.data[i])
                correct += 1;
        }

        std.debug.print("[{}] test_accuracy: {}, cost: {}\n", .{ epoch, @as(f32, @floatFromInt(correct)) / testing_images_count, accumulated_loss / training_images_count });
    }

    const out_file = try std.fs.cwd().createFile(file_name, .{});
    defer out_file.close();

    const t1 = std.time.microTimestamp();

    try Layer.serializeLayers(allocator, classifier.layers, out_file);

    const t2 = std.time.microTimestamp();

    std.debug.print("serialization_time: {} us\n", .{t2 - t1});
}

fn drawImage(data: []const f32) void {
    for (0..28) |i| {
        for (0..28) |j| {
            const character = getBrightnessCharacter(data[i * 28 + j]);
            std.debug.print("{c}{c}", .{ character, character });
        }

        std.debug.print("\n", .{});
    }
}

fn drawNormalizedImage(data: []const f32, mean: []const f32, sd: []const f32, threshold: f32) void {
    for (0..28) |i| {
        for (0..28) |j| {
            const index = i * 28 + j;
            const original_value = std.math.clamp(if (sd[index] >= threshold) (data[index] * sd[index] + mean[index]) else data[index], 0, 1);
            const character = getBrightnessCharacter(original_value);
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
