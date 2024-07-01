#include "shared/type/boolean.h"
#include "shared/context.h"

Boolean::Boolean() : Type("bool") {}

std::string Boolean::get_specifier() const {
    return "%d";
}

mlir::Type Boolean::get_mlir() const {
    return ctx::builder->getI1Type();
}
