CC ?= cc
CFLAGS ?= -O3 -mcpu=native -Wall -Wextra

all: verify bench solver

eval7_tables.h ref5.bin: tables.py eval5.py
	python3 tables.py

verify: verify.c eval7.h eval7_tables.h
	$(CC) $(CFLAGS) -o $@ verify.c -lpthread

bench: bench.c eval7.h eval7_tables.h
	$(CC) $(CFLAGS) -o $@ bench.c

solver: solver.c eval7.h eval7_tables.h
	$(CC) $(CFLAGS) -o $@ solver.c -lpthread -lm

test: verify ref5.bin solver
	python3 test_eval5.py
	python3 test_tables.py
	./verify ref5.bin
	python3 test_solver.py

clean:
	rm -f verify bench solver eval7_tables.h ref5.bin

.PHONY: all test clean
