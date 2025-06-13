rebuild:
	cd build && cmake .. && make -j$(nproc) && cd ..

study:
	cd build/apps/study && ./study && cd ../../../

training:
	cd build/apps/training && ./training && cd ../../../

run-study:
	make rebuild && make study

run-training:
	make rebuild && make training