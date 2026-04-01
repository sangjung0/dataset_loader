from __future__ import annotations

import numpy as np

from typing import Protocol, Any, overload, runtime_checkable, TypeVar
from typing_extensions import Self
from collections.abc import Mapping, MutableMapping, Iterable, Generator

from dataset_loader.protocol.sample_protocol import SampleProtocol

D = TypeVar("D", covariant=True)
S = TypeVar("S", bound=SampleProtocol, covariant=True)


@runtime_checkable
class DatasetProtocol(Protocol[D, S]):
    """
    이 프로토콜은 데이터셋이 가져야 하는 속성과 메서드를 정의합니다.
    데이터셋을 구현할 때 이 프로토콜을 사용하지 않습니다.
    구현할 때는 dataset_loader.base.dataset.Dataset 클래스를 상속하여 구현합니다.
    """

    @property
    def dataset(self) -> D:
        """데이터셋의 정보를 담고있는 객체를 반환하는 속성입니다."""
        ...

    @property
    def args(self) -> MutableMapping[str, Any]:
        """
        데이터셋 인스턴스를 생성하는 데 사용된 인자들을 반환하는 속성입니다.
        __init__ 메서드에 바로 사용할 수 있습니다.

        Returns:
            MutableMapping[str, Any]: 데이터셋 인스턴스를 생성하는 데 사용된 인자들의 딕셔너리입니다. 이 딕셔너리는 데이터셋의 상태를 완전히 설명할 수 있어야 합니다. 예를 들어, 데이터셋이 특정 파일에서 로드된 경우, 파일 경로가 포함되어야 합니다. 데이터셋이 여러 데이터셋을 포함하는 경우, 각 데이터셋의 인자들도 포함되어야 합니다.
        """
        ...

    @property
    def is_cleaned(self) -> bool:
        """
        데이터셋이 메모리에서 정리되었는지 여부를 나타내는 속성입니다.

        Returns:
            bool: 데이터셋이 메모리에서 정리되었는지 여부를 나타내는 불리언 값입니다. True인 경우 데이터셋이 메모리에서 정리되었으며, 더 이상 사용할 수 없음을 나타냅니다. False인 경우 데이터셋이 여전히 메모리에 존재하며 사용할 수 있음을 나타냅니다.
        """
        ...

    @property
    def length(self) -> int:
        """
        데이터셋의 샘플 수를 반환하는 속성입니다.

        Returns:
            int: 데이터셋의 샘플 수를 나타내는 정수입니다. 이 값은 __len__ 메서드와 동일한 값을 반환해야 합니다.
        """
        ...

    def __len__(self) -> int:
        """
        데이터셋의 샘플 수를 반환하는 메서드입니다. length 속성과 동일한 값을 반환해야 합니다.

        Returns:
            int: 데이터셋의 샘플 수를 나타내는 정수입니다. 이 값은 length 속성과 동일한 값을 반환해야 합니다.
        """
        ...

    @property
    def name(self) -> str:
        """
        데이터셋의 이름을 반환하는 속성입니다. 단일 데이터셋인 경우 문자열을 반환하고, 여러 데이터셋을 포함하는 경우 이름 목록을 반환해야 합니다.

        Returns:
            str: 데이터셋의 이름을 나타내는 문자열입니다.
        """
        ...

    def __iter__(self) -> Generator[S, None, None]:
        """
        데이터셋의 샘플을 반복할 수 있도록 하는 메서드입니다.

        Returns:
            Generator[SampleProtocol, None, None]: 데이터셋의 샘플을 반복할 수 있는 제너레이터입니다. 각 샘플은 SampleProtocol을 준수하는 객체여야 합니다.
        """
        ...

    def iter(self) -> Generator[S, None, None]:
        """
        데이터셋의 샘플을 반복할 수 있도록 하는 메서드입니다. __iter__ 메서드와 동일한 기능을 수행해야 합니다.

        Returns:
            Generator[SampleProtocol, None, None]: 데이터셋의 샘플을 반복할 수 있는 제너레이터입니다. 각 샘플은 SampleProtocol을 준수하는 객체여야 합니다.
        """
        ...

    @overload
    def __getitem__(self, key: int) -> S: ...
    @overload
    def __getitem__(self, key: slice | Iterable[int]) -> Self: ...
    def __getitem__(self, key: int | slice | Iterable[int]) -> S | Self:
        """
        key == int -> SampleProtocol \n
        key == slice -> Self (using slice method) \n
        key == Iterable[int] -> Self (using select method) \n

        Args:
            key (int | slice | Iterable[int]): 인덱스, 슬라이스 또는 정수 시퀀스입니다.
        Returns:
            SampleProtocol | Self: 인덱싱된 샘플 또는 데이터셋입니다.
        """

        ...

    @overload
    def getitem(self, key: int) -> S: ...
    @overload
    def getitem(self, key: slice | Iterable[int]) -> Self: ...
    def getitem(self, key: int | slice | Iterable[int]) -> S | Self:
        """
        key == int -> SampleProtocol \n
        key == slice -> Self (using slice method) \n
        key == Iterable[int] -> Self (using select method) \n

        Args:
            key (int | slice | Iterable[int]): 인덱스, 슬라이스 또는 정수 시퀀스입니다.
        Returns:
            SampleProtocol | Self: 인덱싱된 샘플 또는 데이터셋입니다.

        Raises:
            IndexError: 인덱스가 범위를 벗어난 경우
            TypeError: key의 타입이 int, slice, Iterable[int]이 아닌 경우
        """
        ...

    def select(self, indices: Iterable[int]) -> Self:
        """
        데이터셋에서 특정 인덱스에 해당하는 샘플을 선택하여 새로운 데이터셋을 반환하는 메서드입니다.

        Args:
            indices (Sequence[int]): 선택할 샘플의 인덱스 시퀀스입니다.
        Returns:
            Self: 선택된 샘플로 구성된 새로운 데이터셋입니다.
        Raises:
            IndexError: indices 중 하나라도 유효한 인덱스 범위를 벗어난 경우
        """
        ...

    def slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> Self:
        """
        데이터셋에서 특정 범위에 해당하는 샘플을 선택하여 새로운 데이터셋을 반환하는 메서드입니다.
        Args:
            start (int | None): 선택할 샘플의 시작 인덱스입니다. None인 경우 시작 인덱스는 0으로 간주됩니다.
            stop (int | None): 선택할 샘플의 종료 인덱스입니다. None인 경우 종료 인덱스는 데이터셋의 길이로 간주됩니다.
            step (int | None): 선택할 샘플의 간격입니다. None인 경우 간격은 1로 간주됩니다.
        Returns:
            Self: 선택된 샘플로 구성된 새로운 데이터셋입니다.
        Raises:
            IndexError: start 또는 stop이 유효한 인덱스 범위를 벗어난 경우
            ValueError: stop이 start보다 작은 경우
        """
        ...

    def sample(
        self,
        size: int = -1,
        start: int = 0,
        *,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> Self:
        """
        데이터셋에서 무작위로 샘플을 선택하여 새로운 데이터셋을 반환하는 메서드입니다.
        기본값으로 호출된 경우 복사본을 반환합니다.

        Args:
            size (int): 선택할 샘플의 수입니다.
            start (int): 선택할 샘플의 시작 인덱스입니다. 기본값은 0입니다.
            rng (np.random.Generator | np.random.RandomState | None): 무작위 샘플링에 사용할 난수 생성기입니다. None인 경우 무작위 샘플링이 수행되지 않습니다.
        Returns:
            Self: 선택된 샘플로 구성된 새로운 데이터셋입니다.
        Raises:
            IndexError: start가 유효한 인덱스 범위를 벗어난 경우
        """
        ...

    def __add__(self, other: DatasetProtocol[Any, Any]) -> DatasetProtocol[Any, Any]:
        """
        두 데이터셋을 연결하여 새로운 데이터셋을 반환하는 메서드입니다.

        Args:
            other (DatasetProtocol[T]): 연결할 다른 데이터셋입니다.
        Returns:
            DatasetProtocol[T]: 연결된 데이터셋입니다.
        """
        ...

    def concat(self, other: DatasetProtocol[Any, Any]) -> DatasetProtocol[Any, Any]:
        """
        두 데이터셋을 연결하여 새로운 데이터셋을 반환하는 메서드입니다. __add__ 메서드와 동일한 기능을 수행해야 합니다.

        Args:
            other (DatasetProtocol[T]): 연결할 다른 데이터셋입니다.
        Returns:
            DatasetProtocol[T]: 연결된 데이터셋입니다.
        Raises:
            ValueError: self와 other의 task가 다른 경우
            TypeError: other의 타입이 DatasetWrapper 또는 ConcatDataset이 아닌 경우
        """
        ...

    def get(self, idx: int) -> S:
        """
        데이터셋에서 특정 인덱스에 해당하는 샘플을 반환하는 메서드입니다.

        Args:
            idx (int): 반환할 샘플의 인덱스입니다.
        Returns:
            SampleProtocol: 지정된 인덱스에 해당하는 샘플입니다.
        Raises:
            IndexError: idx가 유효한 인덱스 범위를 벗어난 경우
        """
        ...

    def clean(self) -> None:
        """
        데이터셋이 메모리에서 정리되도록 하는 메서드입니다.
        이 메서드를 호출한 후 데이터셋은 더 이상 사용할 수 없습니다.
        """
        ...

    def to_dict(self) -> MutableMapping[str, Any]:
        """
        데이터셋을 설명하는 딕셔너리를 반환하는 메서드입니다. 이 딕셔너리는 데이터셋의 상태를 완전히 설명할 수 있어야 하며, from_dict 클래스 메서드를 사용하여 데이터셋을 dict에서 객체로 복원할 수 있도록 필요한 모든 정보를 포함해야 합니다.
        직렬화 가능하며, from_dict 메서드를 사용하여 복원할 수 있어야 합니다.

        Returns:
            MutableMapping[str, Any]: 데이터셋을 설명하는 딕셔너리입니다. 이 딕셔너리는 데이터셋의 상태를 완전히 설명할 수 있어야 하며, from_dict 클래스 메서드를 사용하여 데이터셋을 dict에서 객체로 복원할 수 있도록 필요한 모든 정보를 포함해야 합니다.
        """
        ...

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """
        데이터셋을 설명하는 딕셔너리에서 데이터셋 객체를 생성하는 클래스 메서드입니다. 이 메서드는 to_dict 메서드에서 반환된 딕셔너리를 사용하여 데이터셋을 복원할 수 있어야 합니다.

        Args:
            data: 데이터셋을 설명하는 딕셔너리입니다. 이 딕셔너리는 to_dict 메서드에서 반환된 딕셔너리를 사용하여 데이터셋을 복원할 수 있어야 합니다.
        Returns:
            Self: 주어진 딕셔너리를 설명하는 데이터셋 객체입니다.
        """
        ...

    def __getstate__(self) -> MutableMapping[str, Any]:
        """
        데이터셋을 직렬화하기 위해 데이터셋을 초기화하기 위한 데이터와 참조하는 클래스를 import하는 데 필요한 정보를 포함하는 딕셔너리를 반환하는 메서드입니다. 이 메서드는 데이터셋을 직렬화할 때 사용됩니다.
        __setstate__ 메서드를 통해서 복원할 수 있습니다.

        Returns:
            MutableMapping[str, Any]: 데이터셋을 초기화하기 위한 데이터와 참조하는 클래스를 import하는 데 필요한 정보를 포함하는 딕셔너리입니다. 이 딕셔너리는 데이터셋을 직렬화할 때 사용됩니다.
        """
        ...

    @classmethod
    def __setstate__(cls, state: Mapping[str, Any]) -> Self:
        """
        데이터셋을 초기화하기 위한 데이터와 참조하는 클래스를 import하는 데 필요한 정보를 포함하는 딕셔너리에서 데이터셋 객체를 생성하는 클래스 메서드입니다. 이 메서드는 __getstate__ 메서드에서 반환된 딕셔너리를 사용하여 데이터셋을 복원할 수 있어야 합니다.

        Args:
            state: 데이터셋을 초기화하기 위한 데이터와 참조하는 클래스를 import하는 데 필요한 정보를 포함하는 딕셔너리입니다. 이 딕셔너리는 __getstate__ 메서드에서 반환된 딕셔너리를 사용하여 데이터셋을 복원할 수 있어야 합니다.

        Returns:
            Self: 주어진 딕셔너리를 설명하는 데이터셋 객체입니다.
        """
        ...

    def __get_import__(self) -> MutableMapping[str, Any]:
        """
        데이터셋을 초기화하기 위해서 참조하는 클래스를 import하는 데 필요한 정보를 반환하는 메서드입니다. 이 메서드는 데이터셋을 직렬화할 때 사용됩니다.

        Returns:
            MutableMapping[str, Any]: 데이터셋을 초기화하기 위해서 참조하는 클래스를 import하는 데 필요한 정보를 포함하는 딕셔너리입니다. 이 딕셔너리는 데이터셋을 직렬화할 때 사용됩니다.
        """
        ...

    @classmethod
    def __set_import__(
        cls, import_info: Mapping[str, Any]
    ) -> tuple[MutableMapping[str, Any], type[DatasetProtocol[D, S]]]:
        """
        데이터셋을 초기화하기 위해서 참조하는 클래스를 import하는 메서드입니다. 이 메서드는 데이터셋을 직렬화할 때 사용됩니다.

        Args:
            import_info: 데이터셋을 초기화하기 위해서 참조하는 클래스를 import하는 데 필요한 정보를 포함하는 딕셔너리입니다. 이 딕셔너리는 데이터셋을 직렬화할 때 사용됩니다.
        Returns:
            tuple[MutableMapping[str, Any], type[DatasetProtocol[T, S]]]: 데이터셋을 초기화하기 위해서 참조하는 클래스를 import하는 데 필요한 정보를 포함하는 딕셔너리와 참조하는 클래스의 타입입니다. 이 메서드는 데이터셋을 직렬화할 때 사용됩니다.
        """
        ...


__all__ = ["DatasetProtocol"]
