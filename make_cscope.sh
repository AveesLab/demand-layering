rm -rf cscope.files cscope.out tags
ctags -R --extra=+f+q --langmap=C:.c.cu.cuh .
find . \( -name '*.c' -o -name '*.cu' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) -print > cscope.files
