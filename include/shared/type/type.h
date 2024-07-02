#pragma once

#include <memory>
#include <string>

#include "mlir/IR/Types.h"

class Type {
   protected:
    std::string name;

   public:
    explicit Type(std::string name);

    std::string get_name() const;
    mlir::Type get_pointer();

    virtual std::string get_specifier() const;
    virtual mlir::Type get_mlir() const;
    virtual bool is_numeric() const;

    bool operator==(const Type rhs) const;
};

class Any : public Type {
   public:
    Any();
    std::string get_specifier() const override;
};

class None : public Type {
   public:
    None();
    std::string get_specifier() const override;
    mlir::Type get_mlir() const override;
};

class Unset : public Type {
   public:
    Unset();
    std::string get_specifier() const override;
};

typedef std::shared_ptr<Type> TypePtr;
