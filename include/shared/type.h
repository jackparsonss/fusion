#pragma once

#include <memory>
#include <string>

class Type {
   private:
    std::string name;

   public:
    explicit Type(std::string name);
    std::string get_name() const;

    bool operator==(Type rhs) const;

    static const Type i32;
    static const Type none;
    static const Type unset;
};

typedef std::shared_ptr<Type> TypePtr;
