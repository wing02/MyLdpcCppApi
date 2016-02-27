CXXFLAGS =	-I /Users/wing/Sdk/eigen  -g -O0

OBJS =		MyLdpc.o 

TEST_OBJS = Test.o MyLdpc.o 

LIBS = -framework OpenCL

TARGET =	MyLdpc

TEST =  MyTest


all:	$(TEST)

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

$(TEST): $(TEST_OBJS)
	$(CXX)  -o $(TEST)  $(TEST_OBJS) $(LIBS)

clean:
	rm -f $(OBJS) $(TARGET) $(TEST) $(TEST_OBJS)
