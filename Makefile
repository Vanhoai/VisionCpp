rebuild:
	cd build && cmake .. && make -j$(nproc) && cd ..

study:
	cd build/apps/study && ./study && cd ../../../

run-study:
	make rebuild && make study