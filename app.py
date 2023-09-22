i m p o r t numpy as np
i m p o r t t e n s o r f l o w as t f
 from t e n s o r f l o w i m p o r t k e r a s
 from t e n s o r f l o w . k e r a s i m p o r t l a y e r s

 # Load t h e d a t a s e t
 ( x t r a i n , y t r a i n ) , ( x t e s t , y t e s t ) = k e r a  . d a t a s e t s . c i f a r 1 0 . l o a d d a t a ( )

 # Normalize t h e p i x e l v a l u e s t o be between 0 and 1
 x t r a i n = x t r a i n . a s t y p e ( ” f l o a t 3 2 ” ) / 255
 x t e s t = x t e s t . a s t y p e ( ” f l o a t 3 2 ” ) / 255

 # D e f i n e t h e model a r c h i t e c t u r e
 model = k e r a s . S e q u e n t i a l (
 [
 l a y e r s . I n p u t ( shape =(32 , 32 , 3) ) ,
 l a y e r s . Conv2D ( 3 2 , k e r n e l s i z e = ( 3 , 3) , a c t i v a t i o n =” r e l u ” ) ,
 l a y e r s . MaxPooling2D ( p o o l s i z e = ( 2 , 2) ) ,
 l a y e r s . Conv2D ( 6 4 , k e r n e l s i z e = ( 3 , 3) , a c t i v a t i o n =” r e l u ” ) ,
 l a y e r s . MaxPooling2D ( p o o l s i z e = ( 2 , 2) ) ,
l a y e r s . F l a t t e n ( ) ,
 l a y e r s . Dense ( 1 0 , a c t i v a t i o n =” softmax ” ) ,
 ]
)

 # Compile t h e model
model . compile ( l o s s =” s p a r s e c a t e g o r i c a l c r o s s e n t r o p y ” , o p t i m i z e r =” adam” , m e t r i c s =[ ” a c c u r a c y ” ] )
 # T r a i n t h e model
model . f i t ( x t r a i n , y t r a i n , b a t c h s i z e =64 , epochs =10 , v a l i d a t i o n s p l i t = 0 . 1 )
 # E v a l u a t e t h e model on t h e t e s t s e t
model . e v a l u a t e ( x t e s t , y t e s t )