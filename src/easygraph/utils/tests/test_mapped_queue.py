import unittest

import easygraph as eg


class MappedQueueTest(unittest.TestCase):
    def test_mapped_queue_push(self):
        mapped_queue = eg.MappedQueue([916, 50, 4609, 493, 237])
        mapped_queue.push(1310)
        self.assertEqual(mapped_queue.h[2], 1310)

    def test_mapped_queue_pop(self):
        mapped_queue = eg.MappedQueue([916, 50, 4609, 493, 237])
        self.assertEqual(mapped_queue.pop(), 50)

    def test_mapped_queue_update(self):
        mapped_queue = eg.MappedQueue([916, 50, 4609, 493, 237])
        mapped_queue.update(916, 100)
        self.assertEqual(mapped_queue.h[1], 100)
        self.assertNotIn(916, mapped_queue.h)

    def test_mapped_queue_remove(self):
        mapped_queue = eg.MappedQueue([916, 50, 4609, 493, 237])
        mapped_queue.remove(237)
        self.assertNotIn(237, mapped_queue.h)
        with self.assertRaises(KeyError):
            mapped_queue.remove(88)

    def test_heapify_duplicate(self):
        with self.assertRaises(AssertionError):
            eg.MappedQueue([916, 916, 50, 4609, 493, 237])

    def test_len(self):
        mapped_queue = eg.MappedQueue([916, 50, 4609, 493, 237])
        self.assertEqual(mapped_queue.__len__(), 5)


if __name__ == "__main__":
    unittest.main()
