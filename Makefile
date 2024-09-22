.PHONY = build run b clean_docker stop all clean fclean

image_name = leaf
container_name = leaf

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
		docker run -t -d --rm --name $(container_name) -e DISPLAY=docker.for.mac.host.internal:0 -v /tmp/.X11-unix:/tmp/.X11-unix -v $(PWD):/work/ $(image_name) /bin/bash; \
	fi

b: run
	docker exec -it $(container_name) /bin/bash

clean_docker: FORCE
#	docker image rm $(image_name) 

stop: FORCE
	docker container stop $(container_name)

FORCE:

# ============= ft_linear_regression ===========================

FLAGS = -Wall -Wextra -Werror -std=c++98
FLAGS += -g
FLAGS += -fsanitize=address

CC = clang++

OBJ_PATH = ./obj
SRC_PATH = ./src

TARGETS = train predict
SRC = $(notdir $(wildcard ./src/*.cpp))
OBJ = $(addprefix $(OBJ_PATH)/, $(SRC:.cpp=.o))

# Default target
all: $(TARGETS)

# Linking the executables
predict: $(OBJ_PATH)/predict.o
	$(CC) $(FLAGS) -o $@ $^

train: $(OBJ_PATH)/train.o
	$(CC) $(FLAGS) -o $@ $^

# Compiling source files to object files
$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cpp
	@mkdir -p $(OBJ_PATH)  # Create the obj directory if it doesn't exist
	$(CC) $(FLAGS) -c $< -o $@

clean:
	echo "removing" $(OBJ_PATH) "..."
	rm -fR $(OBJ_PATH) 

fclean: clean
	rm -f $(TARGETS)
	rm -f theta
	rm -f *.txt

re: fclean all
