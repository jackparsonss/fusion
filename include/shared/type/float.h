#pragma once

#include "shared/type/type.h"

class F32 : public Type {
   public:
    F32();
    std::string get_specifier() const override;
    mlir::Type get_mlir() const override;
};
