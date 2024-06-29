const std = @import("std");

// Slices with len > (std.math.maxInt(usize) / 2) will not have their contents serialized; only their memory occupancy will be taken note of (i.e., as having length (std.math.maxInt(usize) - len))

pub fn serialize(allocator: std.mem.Allocator, object: anytype, file_name: []const u8) !void {
    const file = try std.fs.cwd().createFile(file_name, .{});
    defer file.close();

    var pointer_list = std.ArrayListUnmanaged(usize){};
    defer pointer_list.deinit(allocator);

    try serializeRecursive(allocator, object, file, &pointer_list);
}

pub fn deserialize(comptime T: type, allocator: std.mem.Allocator, file_name: []const u8) !T {
    const file = try std.fs.cwd().openFile(file_name, .{});
    defer file.close();

    var pointer_list = std.ArrayListUnmanaged(usize){};
    defer pointer_list.deinit(allocator);

    return deserializeRecursive(T, allocator, file, &pointer_list);
}

fn serializeRecursive(allocator: std.mem.Allocator, object: anytype, file: std.fs.File, pointer_list: *std.ArrayListUnmanaged(usize)) !void {
    const writer = file.writer();

    switch (@typeInfo(@TypeOf(object))) {
        .Struct => |struct_info| {
            inline for (struct_info.fields) |field_info|
                try serializeRecursive(allocator, @field(object, field_info.name), file, pointer_list);
        },
        .Array => |array_info| switch (@typeInfo(array_info.child)) {
            .Struct, .Array, .Pointer => {
                for (object) |item|
                    try serializeRecursive(allocator, item, file, pointer_list);
            },
            else => try file.writeAll(std.mem.sliceAsBytes(&object)), // Zig 0.12.0 does not implement @ptrCast between slices of different lengths :(
        },
        .Pointer => |pointer_info| {
            switch (pointer_info.size) {
                .Slice => blk: {
                    for (pointer_list.items, 0..) |item, index| {
                        if (item == @intFromPtr(object.ptr)) {
                            try writer.writeInt(usize, index, .little);
                            try writer.writeInt(usize, object.len, .little);

                            break :blk;
                        }
                    }

                    try pointer_list.append(allocator, @intFromPtr(object.ptr));
                    try writer.writeInt(usize, pointer_list.items.len - 1, .little);
                    try writer.writeInt(usize, object.len, .little);

                    if (object.len > std.math.maxInt(usize) / 2)
                        break :blk;

                    switch (@typeInfo(pointer_info.child)) {
                        .Struct, .Array, .Pointer => {
                            for (object) |item|
                                try serializeRecursive(allocator, item, file, pointer_list);
                        },
                        else => try file.writeAll(std.mem.sliceAsBytes(object)), // Zig 0.12.0 does not implement @ptrCast between slices of different lengths :(
                    }
                },
                else => unreachable,
            }
        },
        else => try file.writeAll(std.mem.asBytes(&object)),
    }
}

fn deserializeRecursive(T: type, allocator: std.mem.Allocator, file: std.fs.File, pointer_list: *std.ArrayListUnmanaged(usize)) !T {
    var ret: T = undefined;

    const reader = file.reader();

    switch (@typeInfo(T)) {
        .Struct => |struct_info| {
            inline for (struct_info.fields) |field_info|
                @field(ret, field_info.name) = try deserializeRecursive(field_info.type, allocator, file, pointer_list);
        },
        .Array => |array_info| switch (@typeInfo(array_info.child)) {
            .Struct, .Array, .Pointer => {
                for (&ret) |*item_pointer|
                    item_pointer.* = try deserializeRecursive(array_info.child, allocator, file, pointer_list);
            },
            else => _ = try file.readAll(std.mem.sliceAsBytes(&ret)), // Zig 0.12.0 does not implement @ptrCast between slices of different lengths :(
        },
        .Pointer => |pointer_info| {
            switch (pointer_info.size) {
                .Slice => {
                    const pointer_index = try reader.readInt(usize, .little);
                    const serialized_length = try reader.readInt(usize, .little);

                    const length = if (serialized_length > std.math.maxInt(usize) / 2) (std.math.maxInt(usize) - serialized_length) else serialized_length;

                    if (pointer_index < pointer_list.items.len) {
                        ret.ptr = @ptrFromInt(pointer_list.items[pointer_index]);
                        ret.len = length;
                    } else {
                        ret = try allocator.alloc(pointer_info.child, length);
                        try pointer_list.append(allocator, @intFromPtr(ret.ptr));

                        if (length == serialized_length) {
                            switch (@typeInfo(pointer_info.child)) {
                                .Struct, .Array, .Pointer => {
                                    for (ret) |*item_pointer|
                                        item_pointer.* = try deserializeRecursive(pointer_info.child, allocator, file, pointer_list);
                                },
                                else => _ = try file.readAll(std.mem.sliceAsBytes(ret)), // Zig 0.12.0 does not implement @ptrCast between slices of different lengths :(
                            }
                        }
                    }
                },
                else => unreachable,
            }
        },
        else => _ = try file.readAll(std.mem.asBytes(&ret)),
    }

    return ret;
}
