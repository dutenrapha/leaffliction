.PHONY = build run b lisp py prolog r s clean stop

#container_name = dev-env
path := $(CURDIR)
container_name = $(notdir $(path))
image_name = dev-env

display_mac = -e DISPLAY=docker.for.mac.host.internal:0
#docker run -t -d --rm --name gui -e DISPLAY=docker.for.mac.host.internal:0 -v /tmp/.X11-unix:/tmp/X11-unix -v $(PWD):/work -p:8789:8787 continuumio/anaconda3 /bin/bash
build:
	docker build -t $(image_name) .

run: FORCE
	@if docker inspect --format='{{.State.Running}}' $(container_name) 2>/dev/null | grep -q 'true'; then \
		echo "Container $(container_name) is running"; \
	else \
		echo "Container $(container_name) is not running"; \
		/opt/X11/bin/xhost +; \
		open -a xquartz; \
		docker run -t -d --rm --name $(container_name) $(display_mac) -v /tmp/.X11-unix:/tmp/.X11-unix -v $(PWD):/work/ -v $(PWD)/home/rstudio:/home/rstudio -p 8789:8787 -p 8888:8888 -e DISABLE_AUTH=true $(image_name) /bin/bash; \
	fi

b: run
	docker exec -it $(container_name) /bin/bash

lisp: run
#	docker exec -it $(container_name) rlwrap sbcl
	docker exec -it $(container_name) clisp

py: run
	docker exec -it $(container_name) python3

j: run
	docker exec -it $(container_name) jupyter notebook --NotebookApp.token='' --allow-root --ip 0.0.0.0

prolog: run
	docker exec -it $(container_name) prolog

r: run
	docker exec -it $(container_name) R

s: run
	docker exec -it -e USER=rstudio $(container_name) /usr/sbin/rstudio-server restart 
	open http://localhost:8789


clean: FORCE
	docker image rm $(image_name) 

stop: FORCE
	docker container stop $(container_name)

FORCE:
