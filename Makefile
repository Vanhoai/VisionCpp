format:
	find ./ \( -iname '*.cpp' -o -iname '*.hpp' -o -iname '*.h' \) | xargs clang-format -i -style=file

rebuild:
	cd build && cmake .. -G Ninja && ninja

study:
	make rebuild && cd build/apps/study && ./study

training:
	make rebuild && cd build/apps/training && ./training

benchmark:
	make rebuild && cd build/apps/benchmark && ./benchmark
