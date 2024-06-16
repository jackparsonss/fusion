grammar Fusion;

file: statement* EOF;

statement
    : function
    | declaration
    | block
    ;

declaration:
    qualifier ID COLON type EQ expr SEMI
    ;

block: L_CURLY statement* R_CURLY;

function: FUNCTION ID L_PAREN R_PAREN COLON type block;

expr
    : INT #literalInt
    | ID  #variable
    ;

qualifier: CONST | LET;

type: I32;

// keywords
FUNCTION: 'fn';
CONST: 'const';
LET: 'let';

// symbols
SEMI: ';';
COLON: ':';
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

// literals
INT: [0-9]+;
ID: [a-zA-Z_][a-zA-Z0-9_]*;

// skip whitespace
WS : [ \t\r\n]+ -> skip ;
