rebuild:
	cd build && cmake .. -G Ninja && ninja && cd ..

study:
	cd build/apps/study && ./study && cd ../../../

training:
	cd build/apps/training && ./training && cd ../../../

run-study:
	make rebuild && make study

run-training:
	make rebuild && make training