#include "shared/type/float.h"
#include "shared/context.h"

F32::F32() : Type("f32") {}

std::string F32::get_specifier() const {
    return "%g";
}

mlir::Type F32::get_mlir() const {
    return ctx::builder->getF32Type();
}
