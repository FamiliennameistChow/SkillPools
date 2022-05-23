
/**************************************************************************
 * link_demo.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.05.19
 * 
 * @Description:
 *  链表的使用
 ***************************************************************************/
#include <iostream>


typedef struct LinkNode{
    int val;
    LinkNode *next;

    LinkNode(int v): val(v), next(nullptr){ }
    LinkNode(int v, LinkNode* link): val(v), next(link){ }
    LinkNode(){}
};

using namespace std;


void cross_element(LinkNode *head){
    LinkNode *cur = head;
    std::cout << "{ ";
    while (cur != nullptr){
        std::cout << cur->val << " ";
        cur = cur->next;
    }
    std::cout << "}" << std::endl;
}

int main(){

    // 创建链表

    LinkNode *link5 = new LinkNode(5);
    LinkNode *link4 = new LinkNode(4, link5);
    LinkNode *link3 = new LinkNode(3, link4);
    LinkNode *link2 = new LinkNode(2, link3);
    LinkNode *head = new LinkNode(1, link2);

    cross_element(head);

    // 1. 计算链表有多少元素
    // 设置虚拟头的终止条件是 cur->nest != nullptr
    LinkNode *dummyHead = new LinkNode(0);
    dummyHead->next = head;

    LinkNode *cur = dummyHead;

    int count = 0;
    while (cur->next != nullptr){
        cur = cur ->next;
        count++;
    }
    std::cout << " nums of LinkNode: " <<count << std::endl;

    // 2. 获取第几个元素 index

    LinkNode* cur1 = dummyHead;
    int index = 1;
    while (index--){
        cur1 = cur1->next;
    }
    cur1 = cur1 -> next; // 由于有虚拟头
    std::cout << " index val:" << cur1->val << std::endl;

    // 3. 删除第几个元素 指针应该停留在该元素的上一个元素上 0-index
    int ind = 2;
    LinkNode* cur2 = dummyHead;
    while (ind--){
        cur2 = cur2->next;
    }
    LinkNode *temp = cur2 ->next;
    cur2->next = cur2->next->next;
    delete temp;

    cross_element(head);

    // 4. 在末尾添加元素
    LinkNode *new_end = new LinkNode(10);

    LinkNode *cur3 = dummyHead;
    while (cur3->next != nullptr){
        cur3 = cur3->next;
    }

    cur3->next = new_end;

    cross_element(head);

    // 5. 在第几个元素前插入新元素
    int index_new = 2;
    LinkNode *new_node = new LinkNode(15);
    LinkNode *cur4 = dummyHead;

    //指针停留在该元素之前
    while (index_new--){
        cur4 = cur4->next;
    }
    new_node->next = cur4->next;
    cur4->next = new_node;

    cross_element(head);

    // 6. 在头部插入元素
    LinkNode *new_head = new LinkNode(100);
    new_head->next = dummyHead->next;
    dummyHead->next = new_head;
    cross_element(dummyHead->next);












}

