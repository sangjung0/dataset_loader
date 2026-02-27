from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Generator
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future

from dataset_loader.interface import Sample

T = TypeVar("T", bound=Sample)


class ThreadLoaderMixin(ABC, Generic[T]):
    def thread_iter(
        self,
        *,
        num_workers: int = 4,
        prefetch: int = 8,
        shutdown_wait: bool = False,
    ) -> Generator[T, None, None]:
        """
        ThreadPoolExecutor를 사용하여 샘플을 병렬로 로드하는 제너레이터 메서드입니다. \n

        Args:
            num_workers (int): 사용할 스레드의 수입니다. 기본값은 4입니다.
            prefetch (int): 미리 로드할 샘플의 수입니다. 기본값은 8입니다.
            shutdown_wait (bool): Executor를 종료할 때 작업이 완료될 때까지 기다릴지 여부입니다. 기본값은 False입니다.
        Yields:
            T: 로드된 샘플입니다.
        Raises:
            ValueError: num_workers 또는 prefetch가 양의 정수가 아닌 경우, shutdown_wait가 boolean이 아닌 경우 발생합니다.
        """

        if not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")
        if not isinstance(prefetch, int) or prefetch <= 0:
            raise ValueError("prefetch must be a positive integer")
        if not isinstance(shutdown_wait, bool):
            raise ValueError("shutdown_wait must be a boolean")

        executor = ThreadPoolExecutor(max_workers=num_workers)
        futures: deque[Future[T]] = deque()
        try:
            for sample in self:
                futures.append(executor.submit(self._loader, sample))

                if len(futures) >= prefetch:
                    yield futures.popleft().result()

            while futures:
                yield futures.popleft().result()
        finally:
            for f in futures:
                f.cancel()
            executor.shutdown(wait=shutdown_wait, cancel_futures=True)

    @abstractmethod
    def __iter__(self) -> Generator[T, None, None]:
        raise NotImplementedError("Subclasses must implement __iter__ method")

    @abstractmethod
    def _loader(self, sample: T) -> T:
        raise NotImplementedError("Subclasses must implement _loader method")


__all__ = ["ThreadLoaderMixin"]
