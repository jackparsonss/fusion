#include "shared/type/integer.h"
#include "shared/context.h"

I32::I32() : Type("i32") {}

std::string I32::get_specifier() const {
    return "%d";
}

mlir::Type I32::get_mlir() const {
    return ctx::builder->getI32Type();
}
