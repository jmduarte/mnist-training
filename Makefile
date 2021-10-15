MODEL=mnist_mlp

train:
	python ./$(MODEL).py
.PHONY: train

predict:
	python ./$(MODEL)_pred.py
.PHONY: predict

prune:
	python ./$(MODEL)_prune.py
.PHONY: prune

clean:
	rm -f model/*.h5 model/*.json $(MODEL).orig.py
.PHONY: clean
