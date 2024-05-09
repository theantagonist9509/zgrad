const std = @import("std");

const Matrix = @import("matrix.zig").Matrix;
const Program = @import("program.zig").Program;

const Instruction = Program.Instruction;
const Operation = Instruction.Operation;
const Symbol = Instruction.Symbol;

pub const Mlp = struct {
    program: Program,

    layers: []Layer, // Input to penultimate
    output: Symbol,

    y: Symbol, // Label
    loss: Symbol,

    loss_function: Operation,

    pub fn initializeForTraining(allocator: std.mem.Allocator, layers: []Layer, loss_function: Operation) !Mlp { // tagged union in place of layers slice to allow for loading of pretrianed pretrained parameters for further fine tuning
        var mlp = Mlp{
            .program = .{ .instructions = try allocator.alloc(Instruction, 2 * layers.len + 1) },
            .layers = layers,

            .output = try Symbol.initialize(allocator, layers[layers.len - 1].z.value.row_count, 1),
            .y = try Symbol.initialize(allocator, layers[layers.len - 1].z.value.row_count, 1),

            .loss = try Symbol.initialize(allocator, 1, 1),

            .loss_function = loss_function,
        };

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

        mlp.program.instructions[2 * layers.len] = .{
            .operation = loss_function,
            .inputs = try allocator.alloc(Symbol, 2),
            .output = mlp.loss,
        };
        @memcpy(mlp.program.instructions[2 * layers.len].inputs, &[_]Symbol{ mlp.y, mlp.output });

        return mlp;
    }

    pub fn serialize(self: Mlp, allocator: std.mem.Allocator, file: std.fs.File) !void {
        var pointer_list = std.ArrayList(usize).init(allocator);
        defer pointer_list.deinit();

        try serializeObject(self, file, &pointer_list);
    }

    pub fn deserialize(self: *Mlp, allocator: std.mem.Allocator, file: std.fs.File) !void {
        var pointer_list = std.ArrayListUnmanaged(usize){};
        defer pointer_list.deinit(allocator);

        try deserializeObject(allocator, self, file, &pointer_list);
    }

    fn serializeObject(object: anytype, file: std.fs.File, pointer_list: *std.ArrayList(usize)) !void {
        const writer = file.writer();

        switch (@typeInfo(@TypeOf(object))) {
            .Struct => |struct_info| {
                inline for (struct_info.fields) |field_info|
                    try serializeObject(@field(object, field_info.name), file, pointer_list);
            },
            .Pointer => |pointer_info| {
                switch (pointer_info.size) {
                    .Slice => block: {
                        for (pointer_list.items, 0..) |item, index| {
                            if (item == @intFromPtr(object.ptr)) {
                                try writer.writeInt(usize, index, .little);
                                try writer.writeInt(usize, object.len, .little);
                                break :block;
                            }
                        }

                        try pointer_list.append(@intFromPtr(object.ptr));
                        try writer.writeInt(usize, pointer_list.items.len - 1, .little);
                        try writer.writeInt(usize, object.len, .little);

                        for (object) |item| {
                            switch (@typeInfo(@TypeOf(item))) {
                                .Struct, .Pointer => try serializeObject(item, file, pointer_list),
                                else => try writer.writeAll(std.mem.asBytes(&item)),
                            }
                        }
                    },
                    else => unreachable,
                }
            },
            else => try writer.writeAll(std.mem.asBytes(&object)),
        }
    }

    fn deserializeObject(allocator: std.mem.Allocator, object_pointer: anytype, file: std.fs.File, pointer_list: *std.ArrayListUnmanaged(usize)) !void {
        const reader = file.reader();

        switch (@typeInfo(@TypeOf(object_pointer.*))) {
            .Struct => |struct_info| {
                inline for (struct_info.fields) |field_info|
                    try deserializeObject(allocator, &@field(object_pointer, field_info.name), file, pointer_list);
            },
            .Pointer => |pointer_info| {
                switch (pointer_info.size) {
                    .Slice => {
                        const pointer_index = try reader.readInt(usize, .little);
                        const length = try reader.readInt(usize, .little);

                        if (pointer_index < pointer_list.items.len) {
                            object_pointer.*.ptr = @ptrFromInt(pointer_list.items[pointer_index]);
                            object_pointer.*.len = length;
                        } else {
                            object_pointer.* = try allocator.alloc(@TypeOf(object_pointer.*[0]), length);
                            try pointer_list.append(allocator, @intFromPtr(object_pointer.*.ptr));

                            for (object_pointer.*) |*item_pointer| {
                                switch (@typeInfo(@TypeOf(item_pointer.*))) {
                                    .Struct, .Pointer => try deserializeObject(allocator, item_pointer, file, pointer_list),
                                    else => _ = try reader.readAll(std.mem.asBytes(item_pointer)),
                                }
                            }
                        }
                    },
                    else => unreachable,
                }
            },
            else => _ = try reader.readAll(std.mem.asBytes(object_pointer)),
        }
    }

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
                element.* = 3 * (random.float(f32) - 1) * @sqrt(2 / @as(f32, @floatFromInt(layer.w.value.row_count + layer.w.value.column_count))); // CHECK THIS

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
