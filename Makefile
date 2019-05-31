build:
ifeq ($(shell uname), Darwin)
	cargo rustc --release -- -C link-arg=-undefined -C link-arg=dynamic_lookup
else
	cargo build --release
endif

clean:
	cargo clean
