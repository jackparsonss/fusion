grammar Fusion;

file: statement* EOF;

statement:
    expr SEMI
    ;

expr:
    INT #literalInt
    ;

// symbols
SEMI: ';';

// comments
LINE_COMMENT: '//' .*? ('\n' | EOF) -> skip;
COMMENT: '/*' .*? '*/' -> skip;

// Literals
INT: [0-9]+;

// Skip whitespace
WS : [ \t\r\n]+ -> skip ;
