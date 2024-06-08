pub const core = .{
    .Matrix = @import("matrix.zig").Matrix,
    .Program = @import("program.zig").Program,
};

pub const utilities = @import("utilities.zig");

pub const templates = .{
    .Mlp = @import("mlp.zig").Mlp,
    .Transformer = @import("transformer.zig").Transformer,
};
