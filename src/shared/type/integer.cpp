#include "shared/type/integer.h"
#include "shared/context.h"

I32::I32() : Type("i32") {}

std::string I32::get_specifier() const {
    return "%d";
}

mlir::Type I32::get_mlir() const {
    return ctx::builder->getI32Type();
}

bool I32::is_numeric() const {
    return true;
}

I64::I64() : Type("i64") {}

std::string I64::get_specifier() const {
    return "%ld";
}

mlir::Type I64::get_mlir() const {
    return ctx::builder->getI64Type();
}

bool I64::is_numeric() const {
    return true;
}
