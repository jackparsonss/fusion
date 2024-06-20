grammar Fusion;

file: statement* EOF;

statement
    : function
    | call SEMI
    | declaration
    | block
    | return
    ;

declaration:
    variable EQ expr SEMI
    ;

block: L_CURLY statement* R_CURLY;

function: FUNCTION ID L_PAREN variable? (COMMA variable)* R_PAREN COLON type block;

variable: qualifier ID COLON type;

call: ID L_PAREN expr? (COMMA expr)* R_PAREN;

return: RETURN expr SEMI;

expr
    : call      #callExpr 
    | CHARACTER #literalChar
    | INT       #literalInt
    | ID        #identifier
    ;

qualifier: CONST | LET;

type: I32 | CHAR;

// keywords
RETURN: 'return';
FUNCTION: 'fn';
CONST: 'const';
LET: 'let';

// symbols
SEMI: ';';
COLON: ':';
COMMA: ',';
EQ: '=';
L_PAREN: '(';
R_PAREN: ')';
L_CURLY: '{';
R_CURLY: '}';

// comments
LINE_COMMENT: '//' .*? ('\n' | EOF) -> skip;
COMMENT: '/*' .*? '*/' -> skip;

// types
I32: 'i32';
CHAR: 'ch';

// literals
INT: [0-9]+;
ID: [a-zA-Z_][a-zA-Z0-9_]*;
CHARACTER: '\'' ( '\\\\' | '\\0' | '\\a' | '\\b' | '\\t' | '\\n' | '\\r' | '\\"' | '\\\'' | ~[\\'] ) '\'';

// skip whitespace
WS : [ \t\r\n]+ -> skip ;
