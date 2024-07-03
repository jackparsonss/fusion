# Fusion Lang
A toy language to continue learning antlr4 and mlir/llvm and see where I go

## Setting up environment
If you are on macos then I have made a simple install script which will install all requirements/dependencies needed to develop with antlr and mlir, located [here](https://github.com/jackparsonss/fusion/blob/main/scipts/setup_macos.bash)

## Building
```bash
mkdir build
cd build
cmake ..
make
```

## Running
```bash
cd bin
./fuse <filename>.fuse
```

## Documentation
### Declarations
The `let` keyword signifies that the variable can be later changed.
The `const` keyword signifies that the variable cannot be later changed;
```
let x: i32 = 5;
const y: i32 = x;
```

### Types
- **i32**: 32-bit integer
- **i64**: 64-bit integer
- **ch**: 8-bit ascii character
- **bool**: 1-bit boolean(true/false)

### Functions
```
fn foo(let x: i32, let y: ch): i32 {
    print(x);
    print(y);

    return x + 1;
}

fn main(): i32 {
    foo(5, '\n');
    return 0;
}
```

#### Builtin Functions
##### main
Must be defined by the user, is the entry point to the fusion program
```
fn main(): i32 {
    // do stuff
    return 0;
}
```

##### print
Prints the argument passed to stdout
```
print(5);
```

##### println
Prints the argument passed to stdout with a newline at the end
```
println(100);
```
```
```

#### Arithmetic
- addition: `+`
- subtraction: `-`
- power: `^`
- multiplication: `*`
- division: `/`
- modulus: `%`
- greater than(eq): `>`, `>=`
- less than(eq): `<`, `<=`
- equal: `==`
- not equal: `!=`
- and: `&&`
- or: `||`
```
fn main(): i32 {
    let a: i32 = 5 + 5;
    let s: i32 = 5 - 5;
    let p: i32 = 5 ^ 5;
    let m: i32 = 5 * 5;
    let d: i32 = 5 / 5;
    let r: i32 = 5 % 5;

    let gt: bool = 5 > 4;
    let lt: bool = 4 < 5;
    let gte: bool = 5 >= 5;
    let lte: bool = 5 >= 5;
    let eq: bool = 5 == 5;
    let ne: bool = 5 != 5;
    let o: bool = 5 != 5 || 5 == 5;
    let a: bool = 5 != 5 && 5 == 5;
}
```

#### Conditionals
Just like `if`/`else` statements in most languages
```
let x: i32 = 5;
if(x == 4){
    println(0);
} else if(x == 5) {
    println(10);
} else {
    println(x);
}
```
Conditions **must** be booleans
