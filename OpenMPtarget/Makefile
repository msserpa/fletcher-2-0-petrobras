include ../config.mk
include flags.mk

all:
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c openmptarget_driver.c
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c openmptarget_stuff.c
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c openmptarget_propagate.c
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c openmptarget_insertsource.c

clean:
	rm -f *.o *.a
