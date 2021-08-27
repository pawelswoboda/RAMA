include parallelDefs
BNCHMRK = MST

GLOBAL = IO.h parseCommandLine.h parallel.h sequence.h utils.h graph.h graphIO.h gettime.h

CHECKFILES = serialMST.o $(BNCHMRK)Check.o

$(BNCHMRK)Check : $(CHECKFILES)
	$(PCC) $(PLFLAGS) -o $@ $(CHECKFILES)

%.o : %.C $(GLOBAL)
	$(PCC) $(PCFLAGS) -c $< -o $@

$(GLOBAL) :
	ln -s ../../common/$@ .

clean :
	rm -f $(BNCHMRK)Check *.o

cleansrc :
	make -s clean
	rm -f $(GLOBAL) *.pyc
