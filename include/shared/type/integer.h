#pragma once

#include "shared/type/type.h"

class I32 : public Type {
   public:
    I32();
    std::string get_specifier() const override;
    mlir::Type get_mlir() const override;
};
