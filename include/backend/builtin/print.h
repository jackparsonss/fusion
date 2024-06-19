#pragma once

#include "shared/type.h"

namespace builtin {
void define_all();
void define_print(std::shared_ptr<Type> type);
}  // namespace builtin
