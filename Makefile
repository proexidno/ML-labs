
SOURCES := $(wildcard src/*.c)
OFILES := $(patsubst %.c,%.o,$(SOURCES))
LIBRARY := libzuquic.a

CFLAGS := -Iinclude/
OFLAGS := -Isrc/ -Wall -Werror $(CFLAGS)
OUTFLAGS := -Llib -lzuquic $(CFLAGS)

%: lib/$(LIBRARY) %.c
	gcc -o $@ $^ $(OUTFLAGS)

lib/$(LIBRARY): $(OFILES)
	ar rcs $@ $^

%.o: %.c
	gcc -c $(CFLAGS) -Isrc/ -o $@ $^

clean:
	rm -f $(OFILES) lib/$(LIBRARY)
