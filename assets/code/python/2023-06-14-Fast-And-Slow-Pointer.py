from typing import Optional

class Node():
   def __init__(self, data : int) -> None:
      self.data: int = data
      self.next: Optional[Node] = None

class LinkedList():
   def __init__(self) -> None:
      self.root: Optional[Node] = None

   def insert(self, data : int) -> None:
      ''' insert to the end of the list with new node
      containing data'''
      if self.root is None:
         self.root = Node(data)
         return

      node = self.root
      while node.next is not None:
         node = node.next

      node.next = Node(data)

   def find(self, data) -> Optional[Node]:
      ''' Find a node containing data or return None'''
      node = self.root

      while node:
         if node.data == data:
            return node
         node = node.next

      return None

   def hasCycles(self) -> bool:
      """Detect if the linked list has a cycle."""
      # 0 node 1 node case
      if not self.root or not self.root.next:
         return False
      elif self.root == self.root.next:
         return True

      slow = fast = self.root
      while fast and fast.next:
         slow = slow.next
         fast = fast.next.next

         if slow == fast:
            return True

      return False

   def __repr__(self) -> str:
      """Return a string representation of the linked list."""
      nodes = []
      node = self.root

      seen = {}
      while node:
         nodes.append(str(node.data))
         if seen.get(node) is True:
            # We have a cycle
            break
         seen[node] = True
         node = node.next

      return " -> ".join(nodes)

def main():
   alist = LinkedList()

   for ele in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
      alist.insert(ele)

   print(alist.hasCycles())
   # Let create a cycle
   # Create a cycle by connecting the last node to the second node
   alist.find(10).next = alist.find(2)
   print(alist.hasCycles())

   # now the list looks like this below
   # 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10
   #      ^
   #      |________________________________________|

   print(alist)

if __name__ == '__main__':
   main()
