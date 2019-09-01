MODEL=mnist_mlp
URL=https://github.com/keras-team/keras/raw/master/examples/$(MODEL).py

train:
	python ./$(MODEL).py
.PHONY: train

$(MODEL).orig.py:
	wget $(URL) -O $(MODEL).orig.py

download: $(MODEL).orig.py
.PHONY: download

diff: download
	@vimdiff $(MODEL).py $(MODEL).orig.py
.PHONY: diff

predict:
	python ./$(MODEL)_pred.py
.PHONY: predict

clean:
	rm -f model/*.h5 model/*.json $(MODEL).orig.py
.PHONY: clean
