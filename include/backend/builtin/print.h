#pragma once

#include "shared/type.h"

namespace builtin {
void define_all();
void define_print(TypePtr type);
void define_println(TypePtr type);
}  // namespace builtin
