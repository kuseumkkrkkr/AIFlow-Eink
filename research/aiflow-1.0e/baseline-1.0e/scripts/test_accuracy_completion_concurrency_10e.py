"""동시 완료 요청의 중복 추론과 서로 다른 원본 덮어쓰기를 검사한다."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import time
import unittest
from accuracy_joint_runtime_10e import CompletionCache


class ConcurrentCompletionTests(unittest.TestCase):
    """동일 세션에서는 완료 이벤트를 원자적으로 확정해야 한다."""
    def test_same_event_runs_once(self):
        """동시에 도착한 같은 이벤트도 추론은 한 번만 수행한다."""
        cache=CompletionCache();barrier=Barrier(8);calls=[]
        raw={'strokes':[{'points':[{'x':1,'y':2}]}]}
        def request(_):
            """요청 진입을 맞춰 check-then-write 경합을 재현한다."""
            barrier.wait()
            def predict():
                """느린 추론 사이에 다른 요청이 진입할 틈을 만든다."""
                calls.append(1);time.sleep(.02);return {'raw_latex':'x'}
            return cache.complete('event',raw,{},predict)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results=list(pool.map(request,range(8)))
        self.assertEqual(len(calls),1)
        self.assertTrue(all(r==results[0] for r in results))

    def test_conflicting_concurrent_ink_rejected(self):
        """같은 이벤트 ID에 다른 잉크가 경쟁하면 한 요청만 확정한다."""
        cache=CompletionCache();barrier=Barrier(2)
        def request(x):
            """두 원본의 동시 저장을 유도하고 충돌 거부를 수집한다."""
            barrier.wait()
            def predict():
                """출력은 입력에 대응하지만 원본의 덮어쓰기는 허용하지 않는다."""
                time.sleep(.02);return {'raw_latex':str(x)}
            try:
                cache.complete('event',{'strokes':[{'points':[{'x':x,'y':0}]}]}, {},predict)
                return 'accepted'
            except ValueError:
                return 'rejected'
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(request,[1,2]))
        self.assertCountEqual(results,['accepted','rejected'])


if __name__=='__main__':
    unittest.main()
