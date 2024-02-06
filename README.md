# Ko Dense Passage Retrieval (DPR)

[TmaxEdu-KorDPR](https://github.com/TmaxEdu/KorDPR/tree/master) 레포에 영감을 받아 제작하게 되었습니다.

현재 구현체는
gold passage와 gold negative passage(silver passage)는 기존의 방식을 동일하게 차용합니다.

-   [ ]  chunking 멀티 프로세스 적용하기
-   [ ]  이후에 [Multi-passage BERT: A Globally Normalized BERT Model for Open-domain Question Answering](https://arxiv.org/abs/1908.08167) 방식의 추가적인 gold passage 선정 기법을 탐색합니다
-   [ ]  이후에 dynamic passage length 방식으로 모델을 더 개선해볼 예정입니다.

data는 AIHub에서 받은 데이터들이랑 개인적으로 가지고 있는 데이터들 가지고 할거라서 공개는 힘들듯 합니다.

KLUE랑 korquad등 이것저것 많이 활용해볼 예정입니다.

-   [ ]  [dedup](https://github.com/ChenghaoMou/text-dedup)을 해볼 예정입니다. (대다수의 LLM 논문에서 영감을 받았습니다. passage 및 질문 단위로 dedup하면 중복데이터가 발생하지 않을 걸로 예상됩니다.)
-   [ ]  batch sampler를 중복이 없도록 강제 구현해야 할까요? (고민중)
-   [ ]  BM25를 활용한 hard negative를 구현할 예정입니다.

