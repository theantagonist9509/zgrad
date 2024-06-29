const std = @import("std");

pub fn copyTupleSlice(tuple: anytype, start_index: comptime_int, end_index: comptime_int) TupleSlice(@TypeOf(tuple), start_index, end_index) { // can't slice tuples yet :(
    var ret: TupleSlice(@TypeOf(tuple), start_index, end_index) = undefined;

    inline for (0.., start_index..end_index) |i, j|
        ret[i] = tuple[j];

    return ret;
}

pub fn TupleSlice(T: type, start_index: comptime_int, end_index: comptime_int) type { // can't slice tuples yet :(
    comptime var types: [end_index - start_index]type = undefined;

    for (std.meta.fields(T)[start_index..end_index], 0..) |field_info, i| // TODO why can't i directly iterate over types?
        types[i] = field_info.type;

    return std.meta.Tuple(&types);
}
