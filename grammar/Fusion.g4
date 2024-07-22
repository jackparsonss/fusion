grammar Fusion;

file: topLevel* EOF;

topLevel
    : function
    | declaration SEMI
    ;

statement
    : function
    | call SEMI
    | if
    | loop
    | declaration SEMI
    | assignment SEMI
    | block
    | return
    | CONTINUE SEMI
    | BREAK SEMI
    ;

declaration:
    variable EQUAL expr
    ;

block: L_CURLY statement* R_CURLY;

function: FUNCTION ID L_PAREN variable? (COMMA variable)* R_PAREN COLON type block;

variable: qualifier ID COLON type;

assignment: ID EQUAL expr;

call: ID L_PAREN expr? (COMMA expr)* R_PAREN;

if: IF L_PAREN expr R_PAREN block else?;

loop: FOR L_PAREN declaration SEMI expr SEMI assignment R_PAREN block;

else: ELSE (block | if);

return: RETURN expr SEMI;

expr
    : call                                               #callExpr
    | (op=MINUS | op=BANG) expr                          #unary
    | <assoc='right'> expr CARET expr                    #power
    | expr (op=STAR | op=SLASH | op=MOD | op=DSTAR) expr #mulDivMod
    | expr (op=PLUS | op=MINUS) expr                     #addSub
    | expr (op=GT | op=LT | op=GE | op=LE) expr          #gtLtCond
    | expr (op=EQ | op=NE) expr                          #eqNeCond
    | expr (op=DOR | op=DAND) expr                       #andOrCond
    | BOOLEAN                                            #literalBool
    | CHARACTER                                          #literalChar
    | INT                                                #literalInt
    | ID                                                 #identifier
    ;

qualifier: CONST | LET;

type
    : I32
    | I64 
    | CHAR
    | BOOL
    | NONE
    ;

// keywords
RETURN: 'return';
FUNCTION: 'fn';
CONST: 'const';
LET: 'let';
IF: 'if';
ELSE: 'else';
FOR: 'for';
BREAK: 'break';
CONTINUE: 'continue';

// symbols
SEMI: ';';
COLON: ':';
COMMA: ',';
EQUAL: '=';
L_PAREN: '(';
R_PAREN: ')';
L_CURLY: '{';
R_CURLY: '}';
PLUS: '+';
MINUS: '-';
STAR: '*';
DSTAR: '**';
SLASH: '/';
MOD: '%';
GT: '>';
LT: '<';
GE: '>=';
LE: '<=';
NE: '!=';
EQ: '==';
CARET: '^';
DAND: '&&';
DOR: '||';
BANG: '!';

// comments
LINE_COMMENT: '//' .*? ('\n' | EOF) -> skip;
COMMENT: '/*' .*? '*/' -> skip;

// types
I32: 'i32';
I64: 'i64';
CHAR: 'ch';
BOOL: 'bool';
NONE: 'none';

// literals
BOOLEAN: ('true' | 'false');
INT: [0-9]+;
ID: [a-zA-Z_][a-zA-Z0-9_]*;
CHARACTER: '\'' ( '\\\\' | '\\0' | '\\a' | '\\b' | '\\t' | '\\n' | '\\r' | '\\"' | '\\\'' | ~[\\'] ) '\'';

// skip whitespace
WS : [ \t\r\n]+ -> skip ;
