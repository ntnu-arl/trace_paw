��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.12v2.10.0-76-gfdfc646704c8��
�
Adam/dense_127/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_127/bias/v
{
)Adam/dense_127/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_127/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_127/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/dense_127/kernel/v
�
+Adam/dense_127/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_127/kernel/v*
_output_shapes

:@*
dtype0
�
"Adam/batch_normalization_88/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_88/beta/v
�
6Adam/batch_normalization_88/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_88/beta/v*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_88/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_88/gamma/v
�
7Adam/batch_normalization_88/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_88/gamma/v*
_output_shapes
:@*
dtype0
�
Adam/dense_126/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_126/bias/v
{
)Adam/dense_126/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_126/bias/v*
_output_shapes
:@*
dtype0
�
Adam/dense_126/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_126/kernel/v
�
+Adam/dense_126/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_126/kernel/v*
_output_shapes

:@@*
dtype0
�
"Adam/batch_normalization_87/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_87/beta/v
�
6Adam/batch_normalization_87/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_87/beta/v*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_87/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_87/gamma/v
�
7Adam/batch_normalization_87/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_87/gamma/v*
_output_shapes
:@*
dtype0
�
Adam/dense_125/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_125/bias/v
{
)Adam/dense_125/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_125/bias/v*
_output_shapes
:@*
dtype0
�
Adam/dense_125/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/dense_125/kernel/v
�
+Adam/dense_125/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_125/kernel/v*
_output_shapes

:@*
dtype0
�
"Adam/batch_normalization_86/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_86/beta/v
�
6Adam/batch_normalization_86/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_86/beta/v*
_output_shapes
:*
dtype0
�
#Adam/batch_normalization_86/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_86/gamma/v
�
7Adam/batch_normalization_86/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_86/gamma/v*
_output_shapes
:*
dtype0
�
Adam/dense_124/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_124/bias/v
{
)Adam/dense_124/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_124/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_124/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�**(
shared_nameAdam/dense_124/kernel/v
�
+Adam/dense_124/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_124/kernel/v*
_output_shapes
:	�**
dtype0
�
Adam/dense_127/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_127/bias/m
{
)Adam/dense_127/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_127/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_127/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/dense_127/kernel/m
�
+Adam/dense_127/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_127/kernel/m*
_output_shapes

:@*
dtype0
�
"Adam/batch_normalization_88/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_88/beta/m
�
6Adam/batch_normalization_88/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_88/beta/m*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_88/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_88/gamma/m
�
7Adam/batch_normalization_88/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_88/gamma/m*
_output_shapes
:@*
dtype0
�
Adam/dense_126/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_126/bias/m
{
)Adam/dense_126/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_126/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_126/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_126/kernel/m
�
+Adam/dense_126/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_126/kernel/m*
_output_shapes

:@@*
dtype0
�
"Adam/batch_normalization_87/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_87/beta/m
�
6Adam/batch_normalization_87/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_87/beta/m*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_87/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_87/gamma/m
�
7Adam/batch_normalization_87/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_87/gamma/m*
_output_shapes
:@*
dtype0
�
Adam/dense_125/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_125/bias/m
{
)Adam/dense_125/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_125/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_125/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/dense_125/kernel/m
�
+Adam/dense_125/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_125/kernel/m*
_output_shapes

:@*
dtype0
�
"Adam/batch_normalization_86/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_86/beta/m
�
6Adam/batch_normalization_86/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_86/beta/m*
_output_shapes
:*
dtype0
�
#Adam/batch_normalization_86/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_86/gamma/m
�
7Adam/batch_normalization_86/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_86/gamma/m*
_output_shapes
:*
dtype0
�
Adam/dense_124/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_124/bias/m
{
)Adam/dense_124/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_124/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_124/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�**(
shared_nameAdam/dense_124/kernel/m
�
+Adam/dense_124/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_124/kernel/m*
_output_shapes
:	�**
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
t
dense_127/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_127/bias
m
"dense_127/bias/Read/ReadVariableOpReadVariableOpdense_127/bias*
_output_shapes
:*
dtype0
|
dense_127/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_namedense_127/kernel
u
$dense_127/kernel/Read/ReadVariableOpReadVariableOpdense_127/kernel*
_output_shapes

:@*
dtype0
�
&batch_normalization_88/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_88/moving_variance
�
:batch_normalization_88/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_88/moving_variance*
_output_shapes
:@*
dtype0
�
"batch_normalization_88/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_88/moving_mean
�
6batch_normalization_88/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_88/moving_mean*
_output_shapes
:@*
dtype0
�
batch_normalization_88/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_88/beta
�
/batch_normalization_88/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_88/beta*
_output_shapes
:@*
dtype0
�
batch_normalization_88/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_88/gamma
�
0batch_normalization_88/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_88/gamma*
_output_shapes
:@*
dtype0
t
dense_126/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_126/bias
m
"dense_126/bias/Read/ReadVariableOpReadVariableOpdense_126/bias*
_output_shapes
:@*
dtype0
|
dense_126/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*!
shared_namedense_126/kernel
u
$dense_126/kernel/Read/ReadVariableOpReadVariableOpdense_126/kernel*
_output_shapes

:@@*
dtype0
�
&batch_normalization_87/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_87/moving_variance
�
:batch_normalization_87/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_87/moving_variance*
_output_shapes
:@*
dtype0
�
"batch_normalization_87/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_87/moving_mean
�
6batch_normalization_87/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_87/moving_mean*
_output_shapes
:@*
dtype0
�
batch_normalization_87/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_87/beta
�
/batch_normalization_87/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_87/beta*
_output_shapes
:@*
dtype0
�
batch_normalization_87/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_87/gamma
�
0batch_normalization_87/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_87/gamma*
_output_shapes
:@*
dtype0
t
dense_125/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_125/bias
m
"dense_125/bias/Read/ReadVariableOpReadVariableOpdense_125/bias*
_output_shapes
:@*
dtype0
|
dense_125/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_namedense_125/kernel
u
$dense_125/kernel/Read/ReadVariableOpReadVariableOpdense_125/kernel*
_output_shapes

:@*
dtype0
�
&batch_normalization_86/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_86/moving_variance
�
:batch_normalization_86/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_86/moving_variance*
_output_shapes
:*
dtype0
�
"batch_normalization_86/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_86/moving_mean
�
6batch_normalization_86/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_86/moving_mean*
_output_shapes
:*
dtype0
�
batch_normalization_86/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_86/beta
�
/batch_normalization_86/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_86/beta*
_output_shapes
:*
dtype0
�
batch_normalization_86/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_86/gamma
�
0batch_normalization_86/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_86/gamma*
_output_shapes
:*
dtype0
t
dense_124/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_124/bias
m
"dense_124/bias/Read/ReadVariableOpReadVariableOpdense_124/bias*
_output_shapes
:*
dtype0
}
dense_124/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�**!
shared_namedense_124/kernel
v
$dense_124/kernel/Read/ReadVariableOpReadVariableOpdense_124/kernel*
_output_shapes
:	�**
dtype0
�
serving_default_imagePlaceholder*/
_output_shapes
:���������<Z*
dtype0*$
shape:���������<Z
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_imagedense_124/kerneldense_124/bias&batch_normalization_86/moving_variancebatch_normalization_86/gamma"batch_normalization_86/moving_meanbatch_normalization_86/betadense_125/kerneldense_125/bias&batch_normalization_87/moving_variancebatch_normalization_87/gamma"batch_normalization_87/moving_meanbatch_normalization_87/betadense_126/kerneldense_126/bias&batch_normalization_88/moving_variancebatch_normalization_88/gamma"batch_normalization_88/moving_meanbatch_normalization_88/betadense_127/kerneldense_127/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� */
f*R(
&__inference_signature_wrapper_14104203

NoOpNoOp
�y
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�y
value�yB�y B�x
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias*
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+axis
	,gamma
-beta
.moving_mean
/moving_variance*
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses* 
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses
<_random_generator* 
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

Ckernel
Dbias*
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance*
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses* 
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias*
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
daxis
	egamma
fbeta
gmoving_mean
hmoving_variance*
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses* 
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

ukernel
vbias*
�
#0
$1
,2
-3
.4
/5
C6
D7
L8
M9
N10
O11
\12
]13
e14
f15
g16
h17
u18
v19*
j
#0
$1
,2
-3
C4
D5
L6
M7
\8
]9
e10
f11
u12
v13*

w0
x1
y2* 
�
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
9
trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
	�iter
�beta_1
�beta_2

�decay#m�$m�,m�-m�Cm�Dm�Lm�Mm�\m�]m�em�fm�um�vm�#v�$v�,v�-v�Cv�Dv�Lv�Mv�\v�]v�ev�fv�uv�vv�*

�serving_default* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

#0
$1*

#0
$1*
	
w0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_124/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_124/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
,0
-1
.2
/3*

,0
-1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_86/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_86/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_86/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_86/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 

C0
D1*

C0
D1*
	
x0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_125/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_125/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
L0
M1
N2
O3*

L0
M1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_87/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_87/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_87/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_87/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

\0
]1*

\0
]1*
	
y0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_126/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_126/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
e0
f1
g2
h3*

e0
f1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_88/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_88/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_88/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_88/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

u0
v1*

u0
v1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_127/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_127/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 

�trace_0* 
.
.0
/1
N2
O3
g4
h5*
b
0
1
2
3
4
5
6
7
	8

9
10
11
12*

�0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
w0* 
* 
* 
* 

.0
/1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
x0* 
* 
* 
* 

N0
O1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
y0* 
* 
* 
* 

g0
h1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_124/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_124/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_86/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_86/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_125/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_125/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_87/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_87/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_126/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_126/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_88/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_88/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_127/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_127/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_124/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_124/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_86/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_86/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_125/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_125/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_87/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_87/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_126/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_126/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_88/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_88/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/dense_127/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_127/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
z
StaticRegexFullMatchStaticRegexFullMatchsaver_filename"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
\
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
a
Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
h
SelectSelectStaticRegexFullMatchConst_1Const_2"/device:CPU:**
T0*
_output_shapes
: 
`

StringJoin
StringJoinsaver_filenameSelect"/device:CPU:**
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
value�B�7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices$dense_124/kernel/Read/ReadVariableOp"dense_124/bias/Read/ReadVariableOp0batch_normalization_86/gamma/Read/ReadVariableOp/batch_normalization_86/beta/Read/ReadVariableOp6batch_normalization_86/moving_mean/Read/ReadVariableOp:batch_normalization_86/moving_variance/Read/ReadVariableOp$dense_125/kernel/Read/ReadVariableOp"dense_125/bias/Read/ReadVariableOp0batch_normalization_87/gamma/Read/ReadVariableOp/batch_normalization_87/beta/Read/ReadVariableOp6batch_normalization_87/moving_mean/Read/ReadVariableOp:batch_normalization_87/moving_variance/Read/ReadVariableOp$dense_126/kernel/Read/ReadVariableOp"dense_126/bias/Read/ReadVariableOp0batch_normalization_88/gamma/Read/ReadVariableOp/batch_normalization_88/beta/Read/ReadVariableOp6batch_normalization_88/moving_mean/Read/ReadVariableOp:batch_normalization_88/moving_variance/Read/ReadVariableOp$dense_127/kernel/Read/ReadVariableOp"dense_127/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_124/kernel/m/Read/ReadVariableOp)Adam/dense_124/bias/m/Read/ReadVariableOp7Adam/batch_normalization_86/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_86/beta/m/Read/ReadVariableOp+Adam/dense_125/kernel/m/Read/ReadVariableOp)Adam/dense_125/bias/m/Read/ReadVariableOp7Adam/batch_normalization_87/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_87/beta/m/Read/ReadVariableOp+Adam/dense_126/kernel/m/Read/ReadVariableOp)Adam/dense_126/bias/m/Read/ReadVariableOp7Adam/batch_normalization_88/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_88/beta/m/Read/ReadVariableOp+Adam/dense_127/kernel/m/Read/ReadVariableOp)Adam/dense_127/bias/m/Read/ReadVariableOp+Adam/dense_124/kernel/v/Read/ReadVariableOp)Adam/dense_124/bias/v/Read/ReadVariableOp7Adam/batch_normalization_86/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_86/beta/v/Read/ReadVariableOp+Adam/dense_125/kernel/v/Read/ReadVariableOp)Adam/dense_125/bias/v/Read/ReadVariableOp7Adam/batch_normalization_87/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_87/beta/v/Read/ReadVariableOp+Adam/dense_126/kernel/v/Read/ReadVariableOp)Adam/dense_126/bias/v/Read/ReadVariableOp7Adam/batch_normalization_88/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_88/beta/v/Read/ReadVariableOp+Adam/dense_127/kernel/v/Read/ReadVariableOp)Adam/dense_127/bias/v/Read/ReadVariableOpConst"/device:CPU:0*E
dtypes;
927	
�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
o
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0
i
IdentityIdentitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
value�B�7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*�
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927	
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOpAssignVariableOpdense_124/kernel
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_1AssignVariableOpdense_124/bias
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_2AssignVariableOpbatch_normalization_86/gamma
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
k
AssignVariableOp_3AssignVariableOpbatch_normalization_86/beta
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
r
AssignVariableOp_4AssignVariableOp"batch_normalization_86/moving_mean
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
v
AssignVariableOp_5AssignVariableOp&batch_normalization_86/moving_variance
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_6AssignVariableOpdense_125/kernel
Identity_7"/device:CPU:0*
dtype0
U

Identity_8IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_7AssignVariableOpdense_125/bias
Identity_8"/device:CPU:0*
dtype0
U

Identity_9IdentityRestoreV2:8"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_8AssignVariableOpbatch_normalization_87/gamma
Identity_9"/device:CPU:0*
dtype0
V
Identity_10IdentityRestoreV2:9"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_9AssignVariableOpbatch_normalization_87/betaIdentity_10"/device:CPU:0*
dtype0
W
Identity_11IdentityRestoreV2:10"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_10AssignVariableOp"batch_normalization_87/moving_meanIdentity_11"/device:CPU:0*
dtype0
W
Identity_12IdentityRestoreV2:11"/device:CPU:0*
T0*
_output_shapes
:
x
AssignVariableOp_11AssignVariableOp&batch_normalization_87/moving_varianceIdentity_12"/device:CPU:0*
dtype0
W
Identity_13IdentityRestoreV2:12"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_12AssignVariableOpdense_126/kernelIdentity_13"/device:CPU:0*
dtype0
W
Identity_14IdentityRestoreV2:13"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_13AssignVariableOpdense_126/biasIdentity_14"/device:CPU:0*
dtype0
W
Identity_15IdentityRestoreV2:14"/device:CPU:0*
T0*
_output_shapes
:
n
AssignVariableOp_14AssignVariableOpbatch_normalization_88/gammaIdentity_15"/device:CPU:0*
dtype0
W
Identity_16IdentityRestoreV2:15"/device:CPU:0*
T0*
_output_shapes
:
m
AssignVariableOp_15AssignVariableOpbatch_normalization_88/betaIdentity_16"/device:CPU:0*
dtype0
W
Identity_17IdentityRestoreV2:16"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_16AssignVariableOp"batch_normalization_88/moving_meanIdentity_17"/device:CPU:0*
dtype0
W
Identity_18IdentityRestoreV2:17"/device:CPU:0*
T0*
_output_shapes
:
x
AssignVariableOp_17AssignVariableOp&batch_normalization_88/moving_varianceIdentity_18"/device:CPU:0*
dtype0
W
Identity_19IdentityRestoreV2:18"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_18AssignVariableOpdense_127/kernelIdentity_19"/device:CPU:0*
dtype0
W
Identity_20IdentityRestoreV2:19"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_19AssignVariableOpdense_127/biasIdentity_20"/device:CPU:0*
dtype0
W
Identity_21IdentityRestoreV2:20"/device:CPU:0*
T0	*
_output_shapes
:
[
AssignVariableOp_20AssignVariableOp	Adam/iterIdentity_21"/device:CPU:0*
dtype0	
W
Identity_22IdentityRestoreV2:21"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_21AssignVariableOpAdam/beta_1Identity_22"/device:CPU:0*
dtype0
W
Identity_23IdentityRestoreV2:22"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_22AssignVariableOpAdam/beta_2Identity_23"/device:CPU:0*
dtype0
W
Identity_24IdentityRestoreV2:23"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_23AssignVariableOp
Adam/decayIdentity_24"/device:CPU:0*
dtype0
W
Identity_25IdentityRestoreV2:24"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_24AssignVariableOptotalIdentity_25"/device:CPU:0*
dtype0
W
Identity_26IdentityRestoreV2:25"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_25AssignVariableOpcountIdentity_26"/device:CPU:0*
dtype0
W
Identity_27IdentityRestoreV2:26"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_26AssignVariableOpAdam/dense_124/kernel/mIdentity_27"/device:CPU:0*
dtype0
W
Identity_28IdentityRestoreV2:27"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_27AssignVariableOpAdam/dense_124/bias/mIdentity_28"/device:CPU:0*
dtype0
W
Identity_29IdentityRestoreV2:28"/device:CPU:0*
T0*
_output_shapes
:
u
AssignVariableOp_28AssignVariableOp#Adam/batch_normalization_86/gamma/mIdentity_29"/device:CPU:0*
dtype0
W
Identity_30IdentityRestoreV2:29"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_29AssignVariableOp"Adam/batch_normalization_86/beta/mIdentity_30"/device:CPU:0*
dtype0
W
Identity_31IdentityRestoreV2:30"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_30AssignVariableOpAdam/dense_125/kernel/mIdentity_31"/device:CPU:0*
dtype0
W
Identity_32IdentityRestoreV2:31"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_31AssignVariableOpAdam/dense_125/bias/mIdentity_32"/device:CPU:0*
dtype0
W
Identity_33IdentityRestoreV2:32"/device:CPU:0*
T0*
_output_shapes
:
u
AssignVariableOp_32AssignVariableOp#Adam/batch_normalization_87/gamma/mIdentity_33"/device:CPU:0*
dtype0
W
Identity_34IdentityRestoreV2:33"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_33AssignVariableOp"Adam/batch_normalization_87/beta/mIdentity_34"/device:CPU:0*
dtype0
W
Identity_35IdentityRestoreV2:34"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_34AssignVariableOpAdam/dense_126/kernel/mIdentity_35"/device:CPU:0*
dtype0
W
Identity_36IdentityRestoreV2:35"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_35AssignVariableOpAdam/dense_126/bias/mIdentity_36"/device:CPU:0*
dtype0
W
Identity_37IdentityRestoreV2:36"/device:CPU:0*
T0*
_output_shapes
:
u
AssignVariableOp_36AssignVariableOp#Adam/batch_normalization_88/gamma/mIdentity_37"/device:CPU:0*
dtype0
W
Identity_38IdentityRestoreV2:37"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_37AssignVariableOp"Adam/batch_normalization_88/beta/mIdentity_38"/device:CPU:0*
dtype0
W
Identity_39IdentityRestoreV2:38"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_38AssignVariableOpAdam/dense_127/kernel/mIdentity_39"/device:CPU:0*
dtype0
W
Identity_40IdentityRestoreV2:39"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_39AssignVariableOpAdam/dense_127/bias/mIdentity_40"/device:CPU:0*
dtype0
W
Identity_41IdentityRestoreV2:40"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_40AssignVariableOpAdam/dense_124/kernel/vIdentity_41"/device:CPU:0*
dtype0
W
Identity_42IdentityRestoreV2:41"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_41AssignVariableOpAdam/dense_124/bias/vIdentity_42"/device:CPU:0*
dtype0
W
Identity_43IdentityRestoreV2:42"/device:CPU:0*
T0*
_output_shapes
:
u
AssignVariableOp_42AssignVariableOp#Adam/batch_normalization_86/gamma/vIdentity_43"/device:CPU:0*
dtype0
W
Identity_44IdentityRestoreV2:43"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_43AssignVariableOp"Adam/batch_normalization_86/beta/vIdentity_44"/device:CPU:0*
dtype0
W
Identity_45IdentityRestoreV2:44"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_44AssignVariableOpAdam/dense_125/kernel/vIdentity_45"/device:CPU:0*
dtype0
W
Identity_46IdentityRestoreV2:45"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_45AssignVariableOpAdam/dense_125/bias/vIdentity_46"/device:CPU:0*
dtype0
W
Identity_47IdentityRestoreV2:46"/device:CPU:0*
T0*
_output_shapes
:
u
AssignVariableOp_46AssignVariableOp#Adam/batch_normalization_87/gamma/vIdentity_47"/device:CPU:0*
dtype0
W
Identity_48IdentityRestoreV2:47"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_47AssignVariableOp"Adam/batch_normalization_87/beta/vIdentity_48"/device:CPU:0*
dtype0
W
Identity_49IdentityRestoreV2:48"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_48AssignVariableOpAdam/dense_126/kernel/vIdentity_49"/device:CPU:0*
dtype0
W
Identity_50IdentityRestoreV2:49"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_49AssignVariableOpAdam/dense_126/bias/vIdentity_50"/device:CPU:0*
dtype0
W
Identity_51IdentityRestoreV2:50"/device:CPU:0*
T0*
_output_shapes
:
u
AssignVariableOp_50AssignVariableOp#Adam/batch_normalization_88/gamma/vIdentity_51"/device:CPU:0*
dtype0
W
Identity_52IdentityRestoreV2:51"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_51AssignVariableOp"Adam/batch_normalization_88/beta/vIdentity_52"/device:CPU:0*
dtype0
W
Identity_53IdentityRestoreV2:52"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_52AssignVariableOpAdam/dense_127/kernel/vIdentity_53"/device:CPU:0*
dtype0
W
Identity_54IdentityRestoreV2:53"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_53AssignVariableOpAdam/dense_127/bias/vIdentity_54"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
�	
Identity_55Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: ��
�%
�
T__inference_batch_normalization_87_layer_call_and_return_conditional_losses_14105069

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
L
-__inference_dropout_10_layer_call_fn_14104899

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_dense_126_layer_call_fn_14105093

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2dense_126/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
#dense_126/kernel/Regularizer/L2LossL2Loss:dense_126/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_126/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_126/kernel/Regularizer/mulMul+dense_126/kernel/Regularizer/mul/x:output:0,dense_126/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_126/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
G
+__inference_re_lu_44_layer_call_fn_14104860

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:���������Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
K
-__inference_dropout_10_layer_call_fn_14104870

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_86_layer_call_and_return_conditional_losses_14104821

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
+__inference_model_16_layer_call_fn_14103870	
image;
(dense_124_matmul_readvariableop_resource:	�*7
)dense_124_biasadd_readvariableop_resource:L
>batch_normalization_86_assignmovingavg_readvariableop_resource:N
@batch_normalization_86_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_86_batchnorm_mul_readvariableop_resource:F
8batch_normalization_86_batchnorm_readvariableop_resource::
(dense_125_matmul_readvariableop_resource:@7
)dense_125_biasadd_readvariableop_resource:@L
>batch_normalization_87_assignmovingavg_readvariableop_resource:@N
@batch_normalization_87_assignmovingavg_1_readvariableop_resource:@J
<batch_normalization_87_batchnorm_mul_readvariableop_resource:@F
8batch_normalization_87_batchnorm_readvariableop_resource:@:
(dense_126_matmul_readvariableop_resource:@@7
)dense_126_biasadd_readvariableop_resource:@L
>batch_normalization_88_assignmovingavg_readvariableop_resource:@N
@batch_normalization_88_assignmovingavg_1_readvariableop_resource:@J
<batch_normalization_88_batchnorm_mul_readvariableop_resource:@F
8batch_normalization_88_batchnorm_readvariableop_resource:@:
(dense_127_matmul_readvariableop_resource:@7
)dense_127_biasadd_readvariableop_resource:
identity��&batch_normalization_86/AssignMovingAvg�5batch_normalization_86/AssignMovingAvg/ReadVariableOp�(batch_normalization_86/AssignMovingAvg_1�7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_86/batchnorm/ReadVariableOp�3batch_normalization_86/batchnorm/mul/ReadVariableOp�&batch_normalization_87/AssignMovingAvg�5batch_normalization_87/AssignMovingAvg/ReadVariableOp�(batch_normalization_87/AssignMovingAvg_1�7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_87/batchnorm/ReadVariableOp�3batch_normalization_87/batchnorm/mul/ReadVariableOp�&batch_normalization_88/AssignMovingAvg�5batch_normalization_88/AssignMovingAvg/ReadVariableOp�(batch_normalization_88/AssignMovingAvg_1�7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_88/batchnorm/ReadVariableOp�3batch_normalization_88/batchnorm/mul/ReadVariableOp� dense_124/BiasAdd/ReadVariableOp�dense_124/MatMul/ReadVariableOp�<dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp�2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp� dense_125/BiasAdd/ReadVariableOp�dense_125/MatMul/ReadVariableOp�<dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp�2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp� dense_126/BiasAdd/ReadVariableOp�dense_126/MatMul/ReadVariableOp�<dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp�2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp� dense_127/BiasAdd/ReadVariableOp�dense_127/MatMul/ReadVariableOpa
flatten_39/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  r
flatten_39/ReshapeReshapeimageflatten_39/Const:output:0*
T0*(
_output_shapes
:����������*�
dense_124/MatMul/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
dense_124/MatMulMatMulflatten_39/Reshape:output:0'dense_124/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_124/BiasAdd/ReadVariableOpReadVariableOp)dense_124_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_124/BiasAddBiasAdddense_124/MatMul:product:0(dense_124/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
<dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
-dense_124/dense_124/kernel/Regularizer/L2LossL2LossDdense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: q
,dense_124/dense_124/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
*dense_124/dense_124/kernel/Regularizer/mulMul5dense_124/dense_124/kernel/Regularizer/mul/x:output:06dense_124/dense_124/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5batch_normalization_86/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_86/moments/meanMeandense_124/BiasAdd:output:0>batch_normalization_86/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
+batch_normalization_86/moments/StopGradientStopGradient,batch_normalization_86/moments/mean:output:0*
T0*
_output_shapes

:�
0batch_normalization_86/moments/SquaredDifferenceSquaredDifferencedense_124/BiasAdd:output:04batch_normalization_86/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
9batch_normalization_86/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_86/moments/varianceMean4batch_normalization_86/moments/SquaredDifference:z:0Bbatch_normalization_86/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
&batch_normalization_86/moments/SqueezeSqueeze,batch_normalization_86/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
(batch_normalization_86/moments/Squeeze_1Squeeze0batch_normalization_86/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_86/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_86/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_86_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_86/AssignMovingAvg/subSub=batch_normalization_86/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_86/moments/Squeeze:output:0*
T0*
_output_shapes
:�
*batch_normalization_86/AssignMovingAvg/mulMul.batch_normalization_86/AssignMovingAvg/sub:z:05batch_normalization_86/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
&batch_normalization_86/AssignMovingAvgAssignSubVariableOp>batch_normalization_86_assignmovingavg_readvariableop_resource.batch_normalization_86/AssignMovingAvg/mul:z:06^batch_normalization_86/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_86/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_86/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_86_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_86/AssignMovingAvg_1/subSub?batch_normalization_86/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_86/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
,batch_normalization_86/AssignMovingAvg_1/mulMul0batch_normalization_86/AssignMovingAvg_1/sub:z:07batch_normalization_86/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_86/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_86_assignmovingavg_1_readvariableop_resource0batch_normalization_86/AssignMovingAvg_1/mul:z:08^batch_normalization_86/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_86/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_86/batchnorm/addAddV21batch_normalization_86/moments/Squeeze_1:output:0/batch_normalization_86/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_86/batchnorm/RsqrtRsqrt(batch_normalization_86/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_86/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_86_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_86/batchnorm/mulMul*batch_normalization_86/batchnorm/Rsqrt:y:0;batch_normalization_86/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_86/batchnorm/mul_1Muldense_124/BiasAdd:output:0(batch_normalization_86/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
&batch_normalization_86/batchnorm/mul_2Mul/batch_normalization_86/moments/Squeeze:output:0(batch_normalization_86/batchnorm/mul:z:0*
T0*
_output_shapes
:�
/batch_normalization_86/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_86_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_86/batchnorm/subSub7batch_normalization_86/batchnorm/ReadVariableOp:value:0*batch_normalization_86/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_86/batchnorm/add_1AddV2*batch_normalization_86/batchnorm/mul_1:z:0(batch_normalization_86/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������s
re_lu_44/ReluRelu*batch_normalization_86/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������]
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_10/dropout/MulMulre_lu_44/Relu:activations:0!dropout_10/dropout/Const:output:0*
T0*'
_output_shapes
:���������c
dropout_10/dropout/ShapeShapere_lu_44/Relu:activations:0*
T0*
_output_shapes
:�
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0f
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*'
_output_shapes
:����������
dense_125/MatMul/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_125/MatMulMatMuldropout_10/dropout/Mul_1:z:0'dense_125/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_125/BiasAdd/ReadVariableOpReadVariableOp)dense_125_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_125/BiasAddBiasAdddense_125/MatMul:product:0(dense_125/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
-dense_125/dense_125/kernel/Regularizer/L2LossL2LossDdense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: q
,dense_125/dense_125/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
*dense_125/dense_125/kernel/Regularizer/mulMul5dense_125/dense_125/kernel/Regularizer/mul/x:output:06dense_125/dense_125/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5batch_normalization_87/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_87/moments/meanMeandense_125/BiasAdd:output:0>batch_normalization_87/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
+batch_normalization_87/moments/StopGradientStopGradient,batch_normalization_87/moments/mean:output:0*
T0*
_output_shapes

:@�
0batch_normalization_87/moments/SquaredDifferenceSquaredDifferencedense_125/BiasAdd:output:04batch_normalization_87/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������@�
9batch_normalization_87/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_87/moments/varianceMean4batch_normalization_87/moments/SquaredDifference:z:0Bbatch_normalization_87/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
&batch_normalization_87/moments/SqueezeSqueeze,batch_normalization_87/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 �
(batch_normalization_87/moments/Squeeze_1Squeeze0batch_normalization_87/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 q
,batch_normalization_87/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_87/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_87_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
*batch_normalization_87/AssignMovingAvg/subSub=batch_normalization_87/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_87/moments/Squeeze:output:0*
T0*
_output_shapes
:@�
*batch_normalization_87/AssignMovingAvg/mulMul.batch_normalization_87/AssignMovingAvg/sub:z:05batch_normalization_87/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
&batch_normalization_87/AssignMovingAvgAssignSubVariableOp>batch_normalization_87_assignmovingavg_readvariableop_resource.batch_normalization_87/AssignMovingAvg/mul:z:06^batch_normalization_87/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_87/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_87/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_87_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
,batch_normalization_87/AssignMovingAvg_1/subSub?batch_normalization_87/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_87/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@�
,batch_normalization_87/AssignMovingAvg_1/mulMul0batch_normalization_87/AssignMovingAvg_1/sub:z:07batch_normalization_87/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
(batch_normalization_87/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_87_assignmovingavg_1_readvariableop_resource0batch_normalization_87/AssignMovingAvg_1/mul:z:08^batch_normalization_87/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_87/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_87/batchnorm/addAddV21batch_normalization_87/moments/Squeeze_1:output:0/batch_normalization_87/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_87/batchnorm/RsqrtRsqrt(batch_normalization_87/batchnorm/add:z:0*
T0*
_output_shapes
:@�
3batch_normalization_87/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_87_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_87/batchnorm/mulMul*batch_normalization_87/batchnorm/Rsqrt:y:0;batch_normalization_87/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_87/batchnorm/mul_1Muldense_125/BiasAdd:output:0(batch_normalization_87/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
&batch_normalization_87/batchnorm/mul_2Mul/batch_normalization_87/moments/Squeeze:output:0(batch_normalization_87/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
/batch_normalization_87/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_87_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_87/batchnorm/subSub7batch_normalization_87/batchnorm/ReadVariableOp:value:0*batch_normalization_87/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_87/batchnorm/add_1AddV2*batch_normalization_87/batchnorm/mul_1:z:0(batch_normalization_87/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@s
re_lu_45/ReluRelu*batch_normalization_87/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@_
dropout_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_10/dropout_1/MulMulre_lu_45/Relu:activations:0#dropout_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:���������@e
dropout_10/dropout_1/ShapeShapere_lu_45/Relu:activations:0*
T0*
_output_shapes
:�
1dropout_10/dropout_1/random_uniform/RandomUniformRandomUniform#dropout_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0h
#dropout_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
!dropout_10/dropout_1/GreaterEqualGreaterEqual:dropout_10/dropout_1/random_uniform/RandomUniform:output:0,dropout_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
dropout_10/dropout_1/CastCast%dropout_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
dropout_10/dropout_1/Mul_1Muldropout_10/dropout_1/Mul:z:0dropout_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:���������@�
dense_126/MatMul/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_126/MatMulMatMuldropout_10/dropout_1/Mul_1:z:0'dense_126/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_126/BiasAdd/ReadVariableOpReadVariableOp)dense_126_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_126/BiasAddBiasAdddense_126/MatMul:product:0(dense_126/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
-dense_126/dense_126/kernel/Regularizer/L2LossL2LossDdense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: q
,dense_126/dense_126/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
*dense_126/dense_126/kernel/Regularizer/mulMul5dense_126/dense_126/kernel/Regularizer/mul/x:output:06dense_126/dense_126/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5batch_normalization_88/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_88/moments/meanMeandense_126/BiasAdd:output:0>batch_normalization_88/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
+batch_normalization_88/moments/StopGradientStopGradient,batch_normalization_88/moments/mean:output:0*
T0*
_output_shapes

:@�
0batch_normalization_88/moments/SquaredDifferenceSquaredDifferencedense_126/BiasAdd:output:04batch_normalization_88/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������@�
9batch_normalization_88/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_88/moments/varianceMean4batch_normalization_88/moments/SquaredDifference:z:0Bbatch_normalization_88/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
&batch_normalization_88/moments/SqueezeSqueeze,batch_normalization_88/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 �
(batch_normalization_88/moments/Squeeze_1Squeeze0batch_normalization_88/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 q
,batch_normalization_88/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_88/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_88_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
*batch_normalization_88/AssignMovingAvg/subSub=batch_normalization_88/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_88/moments/Squeeze:output:0*
T0*
_output_shapes
:@�
*batch_normalization_88/AssignMovingAvg/mulMul.batch_normalization_88/AssignMovingAvg/sub:z:05batch_normalization_88/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
&batch_normalization_88/AssignMovingAvgAssignSubVariableOp>batch_normalization_88_assignmovingavg_readvariableop_resource.batch_normalization_88/AssignMovingAvg/mul:z:06^batch_normalization_88/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_88/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_88/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_88_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
,batch_normalization_88/AssignMovingAvg_1/subSub?batch_normalization_88/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_88/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@�
,batch_normalization_88/AssignMovingAvg_1/mulMul0batch_normalization_88/AssignMovingAvg_1/sub:z:07batch_normalization_88/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
(batch_normalization_88/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_88_assignmovingavg_1_readvariableop_resource0batch_normalization_88/AssignMovingAvg_1/mul:z:08^batch_normalization_88/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_88/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_88/batchnorm/addAddV21batch_normalization_88/moments/Squeeze_1:output:0/batch_normalization_88/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_88/batchnorm/RsqrtRsqrt(batch_normalization_88/batchnorm/add:z:0*
T0*
_output_shapes
:@�
3batch_normalization_88/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_88_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_88/batchnorm/mulMul*batch_normalization_88/batchnorm/Rsqrt:y:0;batch_normalization_88/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_88/batchnorm/mul_1Muldense_126/BiasAdd:output:0(batch_normalization_88/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
&batch_normalization_88/batchnorm/mul_2Mul/batch_normalization_88/moments/Squeeze:output:0(batch_normalization_88/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
/batch_normalization_88/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_88_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_88/batchnorm/subSub7batch_normalization_88/batchnorm/ReadVariableOp:value:0*batch_normalization_88/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_88/batchnorm/add_1AddV2*batch_normalization_88/batchnorm/mul_1:z:0(batch_normalization_88/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@s
re_lu_46/ReluRelu*batch_normalization_88/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@�
dense_127/MatMul/ReadVariableOpReadVariableOp(dense_127_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_127/MatMulMatMulre_lu_46/Relu:activations:0'dense_127/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_127/BiasAdd/ReadVariableOpReadVariableOp)dense_127_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_127/BiasAddBiasAdddense_127/MatMul:product:0(dense_127/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
#dense_124/kernel/Regularizer/L2LossL2Loss:dense_124/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_124/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_124/kernel/Regularizer/mulMul+dense_124/kernel/Regularizer/mul/x:output:0,dense_124/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
#dense_125/kernel/Regularizer/L2LossL2Loss:dense_125/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_125/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_125/kernel/Regularizer/mulMul+dense_125/kernel/Regularizer/mul/x:output:0,dense_125/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
#dense_126/kernel/Regularizer/L2LossL2Loss:dense_126/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_126/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_126/kernel/Regularizer/mulMul+dense_126/kernel/Regularizer/mul/x:output:0,dense_126/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_127/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^batch_normalization_86/AssignMovingAvg6^batch_normalization_86/AssignMovingAvg/ReadVariableOp)^batch_normalization_86/AssignMovingAvg_18^batch_normalization_86/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_86/batchnorm/ReadVariableOp4^batch_normalization_86/batchnorm/mul/ReadVariableOp'^batch_normalization_87/AssignMovingAvg6^batch_normalization_87/AssignMovingAvg/ReadVariableOp)^batch_normalization_87/AssignMovingAvg_18^batch_normalization_87/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_87/batchnorm/ReadVariableOp4^batch_normalization_87/batchnorm/mul/ReadVariableOp'^batch_normalization_88/AssignMovingAvg6^batch_normalization_88/AssignMovingAvg/ReadVariableOp)^batch_normalization_88/AssignMovingAvg_18^batch_normalization_88/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_88/batchnorm/ReadVariableOp4^batch_normalization_88/batchnorm/mul/ReadVariableOp!^dense_124/BiasAdd/ReadVariableOp ^dense_124/MatMul/ReadVariableOp=^dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp3^dense_124/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_125/BiasAdd/ReadVariableOp ^dense_125/MatMul/ReadVariableOp=^dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp3^dense_125/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_126/BiasAdd/ReadVariableOp ^dense_126/MatMul/ReadVariableOp=^dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp3^dense_126/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_127/BiasAdd/ReadVariableOp ^dense_127/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������<Z: : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_86/AssignMovingAvg&batch_normalization_86/AssignMovingAvg2n
5batch_normalization_86/AssignMovingAvg/ReadVariableOp5batch_normalization_86/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_86/AssignMovingAvg_1(batch_normalization_86/AssignMovingAvg_12r
7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_86/batchnorm/ReadVariableOp/batch_normalization_86/batchnorm/ReadVariableOp2j
3batch_normalization_86/batchnorm/mul/ReadVariableOp3batch_normalization_86/batchnorm/mul/ReadVariableOp2P
&batch_normalization_87/AssignMovingAvg&batch_normalization_87/AssignMovingAvg2n
5batch_normalization_87/AssignMovingAvg/ReadVariableOp5batch_normalization_87/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_87/AssignMovingAvg_1(batch_normalization_87/AssignMovingAvg_12r
7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_87/batchnorm/ReadVariableOp/batch_normalization_87/batchnorm/ReadVariableOp2j
3batch_normalization_87/batchnorm/mul/ReadVariableOp3batch_normalization_87/batchnorm/mul/ReadVariableOp2P
&batch_normalization_88/AssignMovingAvg&batch_normalization_88/AssignMovingAvg2n
5batch_normalization_88/AssignMovingAvg/ReadVariableOp5batch_normalization_88/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_88/AssignMovingAvg_1(batch_normalization_88/AssignMovingAvg_12r
7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_88/batchnorm/ReadVariableOp/batch_normalization_88/batchnorm/ReadVariableOp2j
3batch_normalization_88/batchnorm/mul/ReadVariableOp3batch_normalization_88/batchnorm/mul/ReadVariableOp2D
 dense_124/BiasAdd/ReadVariableOp dense_124/BiasAdd/ReadVariableOp2B
dense_124/MatMul/ReadVariableOpdense_124/MatMul/ReadVariableOp2|
<dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp<dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2h
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_125/BiasAdd/ReadVariableOp dense_125/BiasAdd/ReadVariableOp2B
dense_125/MatMul/ReadVariableOpdense_125/MatMul/ReadVariableOp2|
<dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp<dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2h
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_126/BiasAdd/ReadVariableOp dense_126/BiasAdd/ReadVariableOp2B
dense_126/MatMul/ReadVariableOpdense_126/MatMul/ReadVariableOp2|
<dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp<dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2h
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_127/BiasAdd/ReadVariableOp dense_127/BiasAdd/ReadVariableOp2B
dense_127/MatMul/ReadVariableOpdense_127/MatMul/ReadVariableOp:V R
/
_output_shapes
:���������<Z

_user_specified_nameimage
�
�
,__inference_dense_124_layer_call_fn_14104733

inputs1
matmul_readvariableop_resource:	�*-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2dense_124/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�**
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
#dense_124/kernel/Regularizer/L2LossL2Loss:dense_124/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_124/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_124/kernel/Regularizer/mulMul+dense_124/kernel/Regularizer/mul/x:output:0,dense_124/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_124/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������*: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:����������*
 
_user_specified_nameinputs
�
G
+__inference_re_lu_45_layer_call_fn_14105074

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:���������@Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
G__inference_dense_127_layer_call_and_return_conditional_losses_14105245

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_87_layer_call_fn_14104981

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
d
H__inference_flatten_39_layer_call_and_return_conditional_losses_14104719

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������*Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������*"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<Z:W S
/
_output_shapes
:���������<Z
 
_user_specified_nameinputs
�	
�
,__inference_dense_127_layer_call_fn_14105235

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
I
-__inference_flatten_39_layer_call_fn_14104713

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������*Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������*"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<Z:W S
/
_output_shapes
:���������<Z
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_1_14105263M
;dense_125_kernel_regularizer_l2loss_readvariableop_resource:@
identity��2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp�
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp;dense_125_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:@*
dtype0�
#dense_125/kernel/Regularizer/L2LossL2Loss:dense_125/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_125/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_125/kernel/Regularizer/mulMul+dense_125/kernel/Regularizer/mul/x:output:0,dense_125/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_125/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^dense_125/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp
�$
�
9__inference_batch_normalization_86_layer_call_fn_14104801

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
T__inference_batch_normalization_88_layer_call_and_return_conditional_losses_14105215

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�%
�
T__inference_batch_normalization_86_layer_call_and_return_conditional_losses_14104855

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
G__inference_dense_124_layer_call_and_return_conditional_losses_14104747

inputs1
matmul_readvariableop_resource:	�*-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2dense_124/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�**
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
#dense_124/kernel/Regularizer/L2LossL2Loss:dense_124/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_124/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_124/kernel/Regularizer/mulMul+dense_124/kernel/Regularizer/mul/x:output:0,dense_124/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_124/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������*: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:����������*
 
_user_specified_nameinputs
��
�
#__inference__wrapped_model_14102496	
imageD
1model_16_dense_124_matmul_readvariableop_resource:	�*@
2model_16_dense_124_biasadd_readvariableop_resource:O
Amodel_16_batch_normalization_86_batchnorm_readvariableop_resource:S
Emodel_16_batch_normalization_86_batchnorm_mul_readvariableop_resource:Q
Cmodel_16_batch_normalization_86_batchnorm_readvariableop_1_resource:Q
Cmodel_16_batch_normalization_86_batchnorm_readvariableop_2_resource:C
1model_16_dense_125_matmul_readvariableop_resource:@@
2model_16_dense_125_biasadd_readvariableop_resource:@O
Amodel_16_batch_normalization_87_batchnorm_readvariableop_resource:@S
Emodel_16_batch_normalization_87_batchnorm_mul_readvariableop_resource:@Q
Cmodel_16_batch_normalization_87_batchnorm_readvariableop_1_resource:@Q
Cmodel_16_batch_normalization_87_batchnorm_readvariableop_2_resource:@C
1model_16_dense_126_matmul_readvariableop_resource:@@@
2model_16_dense_126_biasadd_readvariableop_resource:@O
Amodel_16_batch_normalization_88_batchnorm_readvariableop_resource:@S
Emodel_16_batch_normalization_88_batchnorm_mul_readvariableop_resource:@Q
Cmodel_16_batch_normalization_88_batchnorm_readvariableop_1_resource:@Q
Cmodel_16_batch_normalization_88_batchnorm_readvariableop_2_resource:@C
1model_16_dense_127_matmul_readvariableop_resource:@@
2model_16_dense_127_biasadd_readvariableop_resource:
identity��8model_16/batch_normalization_86/batchnorm/ReadVariableOp�:model_16/batch_normalization_86/batchnorm/ReadVariableOp_1�:model_16/batch_normalization_86/batchnorm/ReadVariableOp_2�<model_16/batch_normalization_86/batchnorm/mul/ReadVariableOp�8model_16/batch_normalization_87/batchnorm/ReadVariableOp�:model_16/batch_normalization_87/batchnorm/ReadVariableOp_1�:model_16/batch_normalization_87/batchnorm/ReadVariableOp_2�<model_16/batch_normalization_87/batchnorm/mul/ReadVariableOp�8model_16/batch_normalization_88/batchnorm/ReadVariableOp�:model_16/batch_normalization_88/batchnorm/ReadVariableOp_1�:model_16/batch_normalization_88/batchnorm/ReadVariableOp_2�<model_16/batch_normalization_88/batchnorm/mul/ReadVariableOp�)model_16/dense_124/BiasAdd/ReadVariableOp�(model_16/dense_124/MatMul/ReadVariableOp�)model_16/dense_125/BiasAdd/ReadVariableOp�(model_16/dense_125/MatMul/ReadVariableOp�)model_16/dense_126/BiasAdd/ReadVariableOp�(model_16/dense_126/MatMul/ReadVariableOp�)model_16/dense_127/BiasAdd/ReadVariableOp�(model_16/dense_127/MatMul/ReadVariableOpj
model_16/flatten_39/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  �
model_16/flatten_39/ReshapeReshapeimage"model_16/flatten_39/Const:output:0*
T0*(
_output_shapes
:����������*�
(model_16/dense_124/MatMul/ReadVariableOpReadVariableOp1model_16_dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
model_16/dense_124/MatMulMatMul$model_16/flatten_39/Reshape:output:00model_16/dense_124/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)model_16/dense_124/BiasAdd/ReadVariableOpReadVariableOp2model_16_dense_124_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_16/dense_124/BiasAddBiasAdd#model_16/dense_124/MatMul:product:01model_16/dense_124/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
8model_16/batch_normalization_86/batchnorm/ReadVariableOpReadVariableOpAmodel_16_batch_normalization_86_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0t
/model_16/batch_normalization_86/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-model_16/batch_normalization_86/batchnorm/addAddV2@model_16/batch_normalization_86/batchnorm/ReadVariableOp:value:08model_16/batch_normalization_86/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
/model_16/batch_normalization_86/batchnorm/RsqrtRsqrt1model_16/batch_normalization_86/batchnorm/add:z:0*
T0*
_output_shapes
:�
<model_16/batch_normalization_86/batchnorm/mul/ReadVariableOpReadVariableOpEmodel_16_batch_normalization_86_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
-model_16/batch_normalization_86/batchnorm/mulMul3model_16/batch_normalization_86/batchnorm/Rsqrt:y:0Dmodel_16/batch_normalization_86/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
/model_16/batch_normalization_86/batchnorm/mul_1Mul#model_16/dense_124/BiasAdd:output:01model_16/batch_normalization_86/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
:model_16/batch_normalization_86/batchnorm/ReadVariableOp_1ReadVariableOpCmodel_16_batch_normalization_86_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
/model_16/batch_normalization_86/batchnorm/mul_2MulBmodel_16/batch_normalization_86/batchnorm/ReadVariableOp_1:value:01model_16/batch_normalization_86/batchnorm/mul:z:0*
T0*
_output_shapes
:�
:model_16/batch_normalization_86/batchnorm/ReadVariableOp_2ReadVariableOpCmodel_16_batch_normalization_86_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
-model_16/batch_normalization_86/batchnorm/subSubBmodel_16/batch_normalization_86/batchnorm/ReadVariableOp_2:value:03model_16/batch_normalization_86/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
/model_16/batch_normalization_86/batchnorm/add_1AddV23model_16/batch_normalization_86/batchnorm/mul_1:z:01model_16/batch_normalization_86/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
model_16/re_lu_44/ReluRelu3model_16/batch_normalization_86/batchnorm/add_1:z:0*
T0*'
_output_shapes
:����������
model_16/dropout_10/IdentityIdentity$model_16/re_lu_44/Relu:activations:0*
T0*'
_output_shapes
:����������
(model_16/dense_125/MatMul/ReadVariableOpReadVariableOp1model_16_dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model_16/dense_125/MatMulMatMul%model_16/dropout_10/Identity:output:00model_16/dense_125/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)model_16/dense_125/BiasAdd/ReadVariableOpReadVariableOp2model_16_dense_125_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_16/dense_125/BiasAddBiasAdd#model_16/dense_125/MatMul:product:01model_16/dense_125/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
8model_16/batch_normalization_87/batchnorm/ReadVariableOpReadVariableOpAmodel_16_batch_normalization_87_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0t
/model_16/batch_normalization_87/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-model_16/batch_normalization_87/batchnorm/addAddV2@model_16/batch_normalization_87/batchnorm/ReadVariableOp:value:08model_16/batch_normalization_87/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
/model_16/batch_normalization_87/batchnorm/RsqrtRsqrt1model_16/batch_normalization_87/batchnorm/add:z:0*
T0*
_output_shapes
:@�
<model_16/batch_normalization_87/batchnorm/mul/ReadVariableOpReadVariableOpEmodel_16_batch_normalization_87_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
-model_16/batch_normalization_87/batchnorm/mulMul3model_16/batch_normalization_87/batchnorm/Rsqrt:y:0Dmodel_16/batch_normalization_87/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
/model_16/batch_normalization_87/batchnorm/mul_1Mul#model_16/dense_125/BiasAdd:output:01model_16/batch_normalization_87/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
:model_16/batch_normalization_87/batchnorm/ReadVariableOp_1ReadVariableOpCmodel_16_batch_normalization_87_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
/model_16/batch_normalization_87/batchnorm/mul_2MulBmodel_16/batch_normalization_87/batchnorm/ReadVariableOp_1:value:01model_16/batch_normalization_87/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
:model_16/batch_normalization_87/batchnorm/ReadVariableOp_2ReadVariableOpCmodel_16_batch_normalization_87_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
-model_16/batch_normalization_87/batchnorm/subSubBmodel_16/batch_normalization_87/batchnorm/ReadVariableOp_2:value:03model_16/batch_normalization_87/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
/model_16/batch_normalization_87/batchnorm/add_1AddV23model_16/batch_normalization_87/batchnorm/mul_1:z:01model_16/batch_normalization_87/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
model_16/re_lu_45/ReluRelu3model_16/batch_normalization_87/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@�
model_16/dropout_10/Identity_1Identity$model_16/re_lu_45/Relu:activations:0*
T0*'
_output_shapes
:���������@�
(model_16/dense_126/MatMul/ReadVariableOpReadVariableOp1model_16_dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model_16/dense_126/MatMulMatMul'model_16/dropout_10/Identity_1:output:00model_16/dense_126/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)model_16/dense_126/BiasAdd/ReadVariableOpReadVariableOp2model_16_dense_126_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_16/dense_126/BiasAddBiasAdd#model_16/dense_126/MatMul:product:01model_16/dense_126/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
8model_16/batch_normalization_88/batchnorm/ReadVariableOpReadVariableOpAmodel_16_batch_normalization_88_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0t
/model_16/batch_normalization_88/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-model_16/batch_normalization_88/batchnorm/addAddV2@model_16/batch_normalization_88/batchnorm/ReadVariableOp:value:08model_16/batch_normalization_88/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
/model_16/batch_normalization_88/batchnorm/RsqrtRsqrt1model_16/batch_normalization_88/batchnorm/add:z:0*
T0*
_output_shapes
:@�
<model_16/batch_normalization_88/batchnorm/mul/ReadVariableOpReadVariableOpEmodel_16_batch_normalization_88_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
-model_16/batch_normalization_88/batchnorm/mulMul3model_16/batch_normalization_88/batchnorm/Rsqrt:y:0Dmodel_16/batch_normalization_88/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
/model_16/batch_normalization_88/batchnorm/mul_1Mul#model_16/dense_126/BiasAdd:output:01model_16/batch_normalization_88/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
:model_16/batch_normalization_88/batchnorm/ReadVariableOp_1ReadVariableOpCmodel_16_batch_normalization_88_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
/model_16/batch_normalization_88/batchnorm/mul_2MulBmodel_16/batch_normalization_88/batchnorm/ReadVariableOp_1:value:01model_16/batch_normalization_88/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
:model_16/batch_normalization_88/batchnorm/ReadVariableOp_2ReadVariableOpCmodel_16_batch_normalization_88_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
-model_16/batch_normalization_88/batchnorm/subSubBmodel_16/batch_normalization_88/batchnorm/ReadVariableOp_2:value:03model_16/batch_normalization_88/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
/model_16/batch_normalization_88/batchnorm/add_1AddV23model_16/batch_normalization_88/batchnorm/mul_1:z:01model_16/batch_normalization_88/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
model_16/re_lu_46/ReluRelu3model_16/batch_normalization_88/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@�
(model_16/dense_127/MatMul/ReadVariableOpReadVariableOp1model_16_dense_127_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model_16/dense_127/MatMulMatMul$model_16/re_lu_46/Relu:activations:00model_16/dense_127/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)model_16/dense_127/BiasAdd/ReadVariableOpReadVariableOp2model_16_dense_127_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_16/dense_127/BiasAddBiasAdd#model_16/dense_127/MatMul:product:01model_16/dense_127/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
IdentityIdentity#model_16/dense_127/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp9^model_16/batch_normalization_86/batchnorm/ReadVariableOp;^model_16/batch_normalization_86/batchnorm/ReadVariableOp_1;^model_16/batch_normalization_86/batchnorm/ReadVariableOp_2=^model_16/batch_normalization_86/batchnorm/mul/ReadVariableOp9^model_16/batch_normalization_87/batchnorm/ReadVariableOp;^model_16/batch_normalization_87/batchnorm/ReadVariableOp_1;^model_16/batch_normalization_87/batchnorm/ReadVariableOp_2=^model_16/batch_normalization_87/batchnorm/mul/ReadVariableOp9^model_16/batch_normalization_88/batchnorm/ReadVariableOp;^model_16/batch_normalization_88/batchnorm/ReadVariableOp_1;^model_16/batch_normalization_88/batchnorm/ReadVariableOp_2=^model_16/batch_normalization_88/batchnorm/mul/ReadVariableOp*^model_16/dense_124/BiasAdd/ReadVariableOp)^model_16/dense_124/MatMul/ReadVariableOp*^model_16/dense_125/BiasAdd/ReadVariableOp)^model_16/dense_125/MatMul/ReadVariableOp*^model_16/dense_126/BiasAdd/ReadVariableOp)^model_16/dense_126/MatMul/ReadVariableOp*^model_16/dense_127/BiasAdd/ReadVariableOp)^model_16/dense_127/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������<Z: : : : : : : : : : : : : : : : : : : : 2t
8model_16/batch_normalization_86/batchnorm/ReadVariableOp8model_16/batch_normalization_86/batchnorm/ReadVariableOp2x
:model_16/batch_normalization_86/batchnorm/ReadVariableOp_1:model_16/batch_normalization_86/batchnorm/ReadVariableOp_12x
:model_16/batch_normalization_86/batchnorm/ReadVariableOp_2:model_16/batch_normalization_86/batchnorm/ReadVariableOp_22|
<model_16/batch_normalization_86/batchnorm/mul/ReadVariableOp<model_16/batch_normalization_86/batchnorm/mul/ReadVariableOp2t
8model_16/batch_normalization_87/batchnorm/ReadVariableOp8model_16/batch_normalization_87/batchnorm/ReadVariableOp2x
:model_16/batch_normalization_87/batchnorm/ReadVariableOp_1:model_16/batch_normalization_87/batchnorm/ReadVariableOp_12x
:model_16/batch_normalization_87/batchnorm/ReadVariableOp_2:model_16/batch_normalization_87/batchnorm/ReadVariableOp_22|
<model_16/batch_normalization_87/batchnorm/mul/ReadVariableOp<model_16/batch_normalization_87/batchnorm/mul/ReadVariableOp2t
8model_16/batch_normalization_88/batchnorm/ReadVariableOp8model_16/batch_normalization_88/batchnorm/ReadVariableOp2x
:model_16/batch_normalization_88/batchnorm/ReadVariableOp_1:model_16/batch_normalization_88/batchnorm/ReadVariableOp_12x
:model_16/batch_normalization_88/batchnorm/ReadVariableOp_2:model_16/batch_normalization_88/batchnorm/ReadVariableOp_22|
<model_16/batch_normalization_88/batchnorm/mul/ReadVariableOp<model_16/batch_normalization_88/batchnorm/mul/ReadVariableOp2V
)model_16/dense_124/BiasAdd/ReadVariableOp)model_16/dense_124/BiasAdd/ReadVariableOp2T
(model_16/dense_124/MatMul/ReadVariableOp(model_16/dense_124/MatMul/ReadVariableOp2V
)model_16/dense_125/BiasAdd/ReadVariableOp)model_16/dense_125/BiasAdd/ReadVariableOp2T
(model_16/dense_125/MatMul/ReadVariableOp(model_16/dense_125/MatMul/ReadVariableOp2V
)model_16/dense_126/BiasAdd/ReadVariableOp)model_16/dense_126/BiasAdd/ReadVariableOp2T
(model_16/dense_126/MatMul/ReadVariableOp(model_16/dense_126/MatMul/ReadVariableOp2V
)model_16/dense_127/BiasAdd/ReadVariableOp)model_16/dense_127/BiasAdd/ReadVariableOp2T
(model_16/dense_127/MatMul/ReadVariableOp(model_16/dense_127/MatMul/ReadVariableOp:V R
/
_output_shapes
:���������<Z

_user_specified_nameimage
�
�
T__inference_batch_normalization_88_layer_call_and_return_conditional_losses_14105181

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
љ
�
+__inference_model_16_layer_call_fn_14102934	
image;
(dense_124_matmul_readvariableop_resource:	�*7
)dense_124_biasadd_readvariableop_resource:F
8batch_normalization_86_batchnorm_readvariableop_resource:J
<batch_normalization_86_batchnorm_mul_readvariableop_resource:H
:batch_normalization_86_batchnorm_readvariableop_1_resource:H
:batch_normalization_86_batchnorm_readvariableop_2_resource::
(dense_125_matmul_readvariableop_resource:@7
)dense_125_biasadd_readvariableop_resource:@F
8batch_normalization_87_batchnorm_readvariableop_resource:@J
<batch_normalization_87_batchnorm_mul_readvariableop_resource:@H
:batch_normalization_87_batchnorm_readvariableop_1_resource:@H
:batch_normalization_87_batchnorm_readvariableop_2_resource:@:
(dense_126_matmul_readvariableop_resource:@@7
)dense_126_biasadd_readvariableop_resource:@F
8batch_normalization_88_batchnorm_readvariableop_resource:@J
<batch_normalization_88_batchnorm_mul_readvariableop_resource:@H
:batch_normalization_88_batchnorm_readvariableop_1_resource:@H
:batch_normalization_88_batchnorm_readvariableop_2_resource:@:
(dense_127_matmul_readvariableop_resource:@7
)dense_127_biasadd_readvariableop_resource:
identity��/batch_normalization_86/batchnorm/ReadVariableOp�1batch_normalization_86/batchnorm/ReadVariableOp_1�1batch_normalization_86/batchnorm/ReadVariableOp_2�3batch_normalization_86/batchnorm/mul/ReadVariableOp�/batch_normalization_87/batchnorm/ReadVariableOp�1batch_normalization_87/batchnorm/ReadVariableOp_1�1batch_normalization_87/batchnorm/ReadVariableOp_2�3batch_normalization_87/batchnorm/mul/ReadVariableOp�/batch_normalization_88/batchnorm/ReadVariableOp�1batch_normalization_88/batchnorm/ReadVariableOp_1�1batch_normalization_88/batchnorm/ReadVariableOp_2�3batch_normalization_88/batchnorm/mul/ReadVariableOp� dense_124/BiasAdd/ReadVariableOp�dense_124/MatMul/ReadVariableOp�<dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp�2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp� dense_125/BiasAdd/ReadVariableOp�dense_125/MatMul/ReadVariableOp�<dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp�2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp� dense_126/BiasAdd/ReadVariableOp�dense_126/MatMul/ReadVariableOp�<dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp�2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp� dense_127/BiasAdd/ReadVariableOp�dense_127/MatMul/ReadVariableOpa
flatten_39/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  r
flatten_39/ReshapeReshapeimageflatten_39/Const:output:0*
T0*(
_output_shapes
:����������*�
dense_124/MatMul/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
dense_124/MatMulMatMulflatten_39/Reshape:output:0'dense_124/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_124/BiasAdd/ReadVariableOpReadVariableOp)dense_124_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_124/BiasAddBiasAdddense_124/MatMul:product:0(dense_124/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
<dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
-dense_124/dense_124/kernel/Regularizer/L2LossL2LossDdense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: q
,dense_124/dense_124/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
*dense_124/dense_124/kernel/Regularizer/mulMul5dense_124/dense_124/kernel/Regularizer/mul/x:output:06dense_124/dense_124/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/batch_normalization_86/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_86_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_86/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_86/batchnorm/addAddV27batch_normalization_86/batchnorm/ReadVariableOp:value:0/batch_normalization_86/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_86/batchnorm/RsqrtRsqrt(batch_normalization_86/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_86/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_86_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_86/batchnorm/mulMul*batch_normalization_86/batchnorm/Rsqrt:y:0;batch_normalization_86/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_86/batchnorm/mul_1Muldense_124/BiasAdd:output:0(batch_normalization_86/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
1batch_normalization_86/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_86_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
&batch_normalization_86/batchnorm/mul_2Mul9batch_normalization_86/batchnorm/ReadVariableOp_1:value:0(batch_normalization_86/batchnorm/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_86/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_86_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
$batch_normalization_86/batchnorm/subSub9batch_normalization_86/batchnorm/ReadVariableOp_2:value:0*batch_normalization_86/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_86/batchnorm/add_1AddV2*batch_normalization_86/batchnorm/mul_1:z:0(batch_normalization_86/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������s
re_lu_44/ReluRelu*batch_normalization_86/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������n
dropout_10/IdentityIdentityre_lu_44/Relu:activations:0*
T0*'
_output_shapes
:����������
dense_125/MatMul/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_125/MatMulMatMuldropout_10/Identity:output:0'dense_125/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_125/BiasAdd/ReadVariableOpReadVariableOp)dense_125_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_125/BiasAddBiasAdddense_125/MatMul:product:0(dense_125/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
-dense_125/dense_125/kernel/Regularizer/L2LossL2LossDdense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: q
,dense_125/dense_125/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
*dense_125/dense_125/kernel/Regularizer/mulMul5dense_125/dense_125/kernel/Regularizer/mul/x:output:06dense_125/dense_125/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/batch_normalization_87/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_87_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0k
&batch_normalization_87/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_87/batchnorm/addAddV27batch_normalization_87/batchnorm/ReadVariableOp:value:0/batch_normalization_87/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_87/batchnorm/RsqrtRsqrt(batch_normalization_87/batchnorm/add:z:0*
T0*
_output_shapes
:@�
3batch_normalization_87/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_87_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_87/batchnorm/mulMul*batch_normalization_87/batchnorm/Rsqrt:y:0;batch_normalization_87/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_87/batchnorm/mul_1Muldense_125/BiasAdd:output:0(batch_normalization_87/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
1batch_normalization_87/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_87_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_87/batchnorm/mul_2Mul9batch_normalization_87/batchnorm/ReadVariableOp_1:value:0(batch_normalization_87/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
1batch_normalization_87/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_87_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_87/batchnorm/subSub9batch_normalization_87/batchnorm/ReadVariableOp_2:value:0*batch_normalization_87/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_87/batchnorm/add_1AddV2*batch_normalization_87/batchnorm/mul_1:z:0(batch_normalization_87/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@s
re_lu_45/ReluRelu*batch_normalization_87/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@p
dropout_10/Identity_1Identityre_lu_45/Relu:activations:0*
T0*'
_output_shapes
:���������@�
dense_126/MatMul/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_126/MatMulMatMuldropout_10/Identity_1:output:0'dense_126/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_126/BiasAdd/ReadVariableOpReadVariableOp)dense_126_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_126/BiasAddBiasAdddense_126/MatMul:product:0(dense_126/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
-dense_126/dense_126/kernel/Regularizer/L2LossL2LossDdense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: q
,dense_126/dense_126/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
*dense_126/dense_126/kernel/Regularizer/mulMul5dense_126/dense_126/kernel/Regularizer/mul/x:output:06dense_126/dense_126/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/batch_normalization_88/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_88_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0k
&batch_normalization_88/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_88/batchnorm/addAddV27batch_normalization_88/batchnorm/ReadVariableOp:value:0/batch_normalization_88/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_88/batchnorm/RsqrtRsqrt(batch_normalization_88/batchnorm/add:z:0*
T0*
_output_shapes
:@�
3batch_normalization_88/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_88_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_88/batchnorm/mulMul*batch_normalization_88/batchnorm/Rsqrt:y:0;batch_normalization_88/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_88/batchnorm/mul_1Muldense_126/BiasAdd:output:0(batch_normalization_88/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
1batch_normalization_88/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_88_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_88/batchnorm/mul_2Mul9batch_normalization_88/batchnorm/ReadVariableOp_1:value:0(batch_normalization_88/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
1batch_normalization_88/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_88_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_88/batchnorm/subSub9batch_normalization_88/batchnorm/ReadVariableOp_2:value:0*batch_normalization_88/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_88/batchnorm/add_1AddV2*batch_normalization_88/batchnorm/mul_1:z:0(batch_normalization_88/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@s
re_lu_46/ReluRelu*batch_normalization_88/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@�
dense_127/MatMul/ReadVariableOpReadVariableOp(dense_127_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_127/MatMulMatMulre_lu_46/Relu:activations:0'dense_127/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_127/BiasAdd/ReadVariableOpReadVariableOp)dense_127_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_127/BiasAddBiasAdddense_127/MatMul:product:0(dense_127/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
#dense_124/kernel/Regularizer/L2LossL2Loss:dense_124/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_124/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_124/kernel/Regularizer/mulMul+dense_124/kernel/Regularizer/mul/x:output:0,dense_124/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
#dense_125/kernel/Regularizer/L2LossL2Loss:dense_125/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_125/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_125/kernel/Regularizer/mulMul+dense_125/kernel/Regularizer/mul/x:output:0,dense_125/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
#dense_126/kernel/Regularizer/L2LossL2Loss:dense_126/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_126/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_126/kernel/Regularizer/mulMul+dense_126/kernel/Regularizer/mul/x:output:0,dense_126/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_127/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������

NoOpNoOp0^batch_normalization_86/batchnorm/ReadVariableOp2^batch_normalization_86/batchnorm/ReadVariableOp_12^batch_normalization_86/batchnorm/ReadVariableOp_24^batch_normalization_86/batchnorm/mul/ReadVariableOp0^batch_normalization_87/batchnorm/ReadVariableOp2^batch_normalization_87/batchnorm/ReadVariableOp_12^batch_normalization_87/batchnorm/ReadVariableOp_24^batch_normalization_87/batchnorm/mul/ReadVariableOp0^batch_normalization_88/batchnorm/ReadVariableOp2^batch_normalization_88/batchnorm/ReadVariableOp_12^batch_normalization_88/batchnorm/ReadVariableOp_24^batch_normalization_88/batchnorm/mul/ReadVariableOp!^dense_124/BiasAdd/ReadVariableOp ^dense_124/MatMul/ReadVariableOp=^dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp3^dense_124/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_125/BiasAdd/ReadVariableOp ^dense_125/MatMul/ReadVariableOp=^dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp3^dense_125/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_126/BiasAdd/ReadVariableOp ^dense_126/MatMul/ReadVariableOp=^dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp3^dense_126/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_127/BiasAdd/ReadVariableOp ^dense_127/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������<Z: : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_86/batchnorm/ReadVariableOp/batch_normalization_86/batchnorm/ReadVariableOp2f
1batch_normalization_86/batchnorm/ReadVariableOp_11batch_normalization_86/batchnorm/ReadVariableOp_12f
1batch_normalization_86/batchnorm/ReadVariableOp_21batch_normalization_86/batchnorm/ReadVariableOp_22j
3batch_normalization_86/batchnorm/mul/ReadVariableOp3batch_normalization_86/batchnorm/mul/ReadVariableOp2b
/batch_normalization_87/batchnorm/ReadVariableOp/batch_normalization_87/batchnorm/ReadVariableOp2f
1batch_normalization_87/batchnorm/ReadVariableOp_11batch_normalization_87/batchnorm/ReadVariableOp_12f
1batch_normalization_87/batchnorm/ReadVariableOp_21batch_normalization_87/batchnorm/ReadVariableOp_22j
3batch_normalization_87/batchnorm/mul/ReadVariableOp3batch_normalization_87/batchnorm/mul/ReadVariableOp2b
/batch_normalization_88/batchnorm/ReadVariableOp/batch_normalization_88/batchnorm/ReadVariableOp2f
1batch_normalization_88/batchnorm/ReadVariableOp_11batch_normalization_88/batchnorm/ReadVariableOp_12f
1batch_normalization_88/batchnorm/ReadVariableOp_21batch_normalization_88/batchnorm/ReadVariableOp_22j
3batch_normalization_88/batchnorm/mul/ReadVariableOp3batch_normalization_88/batchnorm/mul/ReadVariableOp2D
 dense_124/BiasAdd/ReadVariableOp dense_124/BiasAdd/ReadVariableOp2B
dense_124/MatMul/ReadVariableOpdense_124/MatMul/ReadVariableOp2|
<dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp<dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2h
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_125/BiasAdd/ReadVariableOp dense_125/BiasAdd/ReadVariableOp2B
dense_125/MatMul/ReadVariableOpdense_125/MatMul/ReadVariableOp2|
<dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp<dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2h
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_126/BiasAdd/ReadVariableOp dense_126/BiasAdd/ReadVariableOp2B
dense_126/MatMul/ReadVariableOpdense_126/MatMul/ReadVariableOp2|
<dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp<dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2h
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_127/BiasAdd/ReadVariableOp dense_127/BiasAdd/ReadVariableOp2B
dense_127/MatMul/ReadVariableOpdense_127/MatMul/ReadVariableOp:V R
/
_output_shapes
:���������<Z

_user_specified_nameimage
�
K
-__inference_dropout_10_layer_call_fn_14104887

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
F__inference_model_16_layer_call_and_return_conditional_losses_14104556

inputs;
(dense_124_matmul_readvariableop_resource:	�*7
)dense_124_biasadd_readvariableop_resource:F
8batch_normalization_86_batchnorm_readvariableop_resource:J
<batch_normalization_86_batchnorm_mul_readvariableop_resource:H
:batch_normalization_86_batchnorm_readvariableop_1_resource:H
:batch_normalization_86_batchnorm_readvariableop_2_resource::
(dense_125_matmul_readvariableop_resource:@7
)dense_125_biasadd_readvariableop_resource:@F
8batch_normalization_87_batchnorm_readvariableop_resource:@J
<batch_normalization_87_batchnorm_mul_readvariableop_resource:@H
:batch_normalization_87_batchnorm_readvariableop_1_resource:@H
:batch_normalization_87_batchnorm_readvariableop_2_resource:@:
(dense_126_matmul_readvariableop_resource:@@7
)dense_126_biasadd_readvariableop_resource:@F
8batch_normalization_88_batchnorm_readvariableop_resource:@J
<batch_normalization_88_batchnorm_mul_readvariableop_resource:@H
:batch_normalization_88_batchnorm_readvariableop_1_resource:@H
:batch_normalization_88_batchnorm_readvariableop_2_resource:@:
(dense_127_matmul_readvariableop_resource:@7
)dense_127_biasadd_readvariableop_resource:
identity��/batch_normalization_86/batchnorm/ReadVariableOp�1batch_normalization_86/batchnorm/ReadVariableOp_1�1batch_normalization_86/batchnorm/ReadVariableOp_2�3batch_normalization_86/batchnorm/mul/ReadVariableOp�/batch_normalization_87/batchnorm/ReadVariableOp�1batch_normalization_87/batchnorm/ReadVariableOp_1�1batch_normalization_87/batchnorm/ReadVariableOp_2�3batch_normalization_87/batchnorm/mul/ReadVariableOp�/batch_normalization_88/batchnorm/ReadVariableOp�1batch_normalization_88/batchnorm/ReadVariableOp_1�1batch_normalization_88/batchnorm/ReadVariableOp_2�3batch_normalization_88/batchnorm/mul/ReadVariableOp� dense_124/BiasAdd/ReadVariableOp�dense_124/MatMul/ReadVariableOp�2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp� dense_125/BiasAdd/ReadVariableOp�dense_125/MatMul/ReadVariableOp�2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp� dense_126/BiasAdd/ReadVariableOp�dense_126/MatMul/ReadVariableOp�2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp� dense_127/BiasAdd/ReadVariableOp�dense_127/MatMul/ReadVariableOpa
flatten_39/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  s
flatten_39/ReshapeReshapeinputsflatten_39/Const:output:0*
T0*(
_output_shapes
:����������*�
dense_124/MatMul/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
dense_124/MatMulMatMulflatten_39/Reshape:output:0'dense_124/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_124/BiasAdd/ReadVariableOpReadVariableOp)dense_124_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_124/BiasAddBiasAdddense_124/MatMul:product:0(dense_124/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/batch_normalization_86/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_86_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_86/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_86/batchnorm/addAddV27batch_normalization_86/batchnorm/ReadVariableOp:value:0/batch_normalization_86/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_86/batchnorm/RsqrtRsqrt(batch_normalization_86/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_86/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_86_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_86/batchnorm/mulMul*batch_normalization_86/batchnorm/Rsqrt:y:0;batch_normalization_86/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_86/batchnorm/mul_1Muldense_124/BiasAdd:output:0(batch_normalization_86/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
1batch_normalization_86/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_86_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
&batch_normalization_86/batchnorm/mul_2Mul9batch_normalization_86/batchnorm/ReadVariableOp_1:value:0(batch_normalization_86/batchnorm/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_86/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_86_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
$batch_normalization_86/batchnorm/subSub9batch_normalization_86/batchnorm/ReadVariableOp_2:value:0*batch_normalization_86/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_86/batchnorm/add_1AddV2*batch_normalization_86/batchnorm/mul_1:z:0(batch_normalization_86/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������s
re_lu_44/ReluRelu*batch_normalization_86/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������n
dropout_10/IdentityIdentityre_lu_44/Relu:activations:0*
T0*'
_output_shapes
:����������
dense_125/MatMul/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_125/MatMulMatMuldropout_10/Identity:output:0'dense_125/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_125/BiasAdd/ReadVariableOpReadVariableOp)dense_125_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_125/BiasAddBiasAdddense_125/MatMul:product:0(dense_125/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/batch_normalization_87/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_87_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0k
&batch_normalization_87/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_87/batchnorm/addAddV27batch_normalization_87/batchnorm/ReadVariableOp:value:0/batch_normalization_87/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_87/batchnorm/RsqrtRsqrt(batch_normalization_87/batchnorm/add:z:0*
T0*
_output_shapes
:@�
3batch_normalization_87/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_87_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_87/batchnorm/mulMul*batch_normalization_87/batchnorm/Rsqrt:y:0;batch_normalization_87/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_87/batchnorm/mul_1Muldense_125/BiasAdd:output:0(batch_normalization_87/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
1batch_normalization_87/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_87_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_87/batchnorm/mul_2Mul9batch_normalization_87/batchnorm/ReadVariableOp_1:value:0(batch_normalization_87/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
1batch_normalization_87/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_87_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_87/batchnorm/subSub9batch_normalization_87/batchnorm/ReadVariableOp_2:value:0*batch_normalization_87/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_87/batchnorm/add_1AddV2*batch_normalization_87/batchnorm/mul_1:z:0(batch_normalization_87/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@s
re_lu_45/ReluRelu*batch_normalization_87/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@p
dropout_10/Identity_1Identityre_lu_45/Relu:activations:0*
T0*'
_output_shapes
:���������@�
dense_126/MatMul/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_126/MatMulMatMuldropout_10/Identity_1:output:0'dense_126/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_126/BiasAdd/ReadVariableOpReadVariableOp)dense_126_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_126/BiasAddBiasAdddense_126/MatMul:product:0(dense_126/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/batch_normalization_88/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_88_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0k
&batch_normalization_88/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_88/batchnorm/addAddV27batch_normalization_88/batchnorm/ReadVariableOp:value:0/batch_normalization_88/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_88/batchnorm/RsqrtRsqrt(batch_normalization_88/batchnorm/add:z:0*
T0*
_output_shapes
:@�
3batch_normalization_88/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_88_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_88/batchnorm/mulMul*batch_normalization_88/batchnorm/Rsqrt:y:0;batch_normalization_88/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_88/batchnorm/mul_1Muldense_126/BiasAdd:output:0(batch_normalization_88/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
1batch_normalization_88/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_88_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_88/batchnorm/mul_2Mul9batch_normalization_88/batchnorm/ReadVariableOp_1:value:0(batch_normalization_88/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
1batch_normalization_88/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_88_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_88/batchnorm/subSub9batch_normalization_88/batchnorm/ReadVariableOp_2:value:0*batch_normalization_88/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_88/batchnorm/add_1AddV2*batch_normalization_88/batchnorm/mul_1:z:0(batch_normalization_88/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@s
re_lu_46/ReluRelu*batch_normalization_88/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@�
dense_127/MatMul/ReadVariableOpReadVariableOp(dense_127_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_127/MatMulMatMulre_lu_46/Relu:activations:0'dense_127/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_127/BiasAdd/ReadVariableOpReadVariableOp)dense_127_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_127/BiasAddBiasAdddense_127/MatMul:product:0(dense_127/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
#dense_124/kernel/Regularizer/L2LossL2Loss:dense_124/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_124/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_124/kernel/Regularizer/mulMul+dense_124/kernel/Regularizer/mul/x:output:0,dense_124/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
#dense_125/kernel/Regularizer/L2LossL2Loss:dense_125/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_125/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_125/kernel/Regularizer/mulMul+dense_125/kernel/Regularizer/mul/x:output:0,dense_125/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
#dense_126/kernel/Regularizer/L2LossL2Loss:dense_126/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_126/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_126/kernel/Regularizer/mulMul+dense_126/kernel/Regularizer/mul/x:output:0,dense_126/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_127/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp0^batch_normalization_86/batchnorm/ReadVariableOp2^batch_normalization_86/batchnorm/ReadVariableOp_12^batch_normalization_86/batchnorm/ReadVariableOp_24^batch_normalization_86/batchnorm/mul/ReadVariableOp0^batch_normalization_87/batchnorm/ReadVariableOp2^batch_normalization_87/batchnorm/ReadVariableOp_12^batch_normalization_87/batchnorm/ReadVariableOp_24^batch_normalization_87/batchnorm/mul/ReadVariableOp0^batch_normalization_88/batchnorm/ReadVariableOp2^batch_normalization_88/batchnorm/ReadVariableOp_12^batch_normalization_88/batchnorm/ReadVariableOp_24^batch_normalization_88/batchnorm/mul/ReadVariableOp!^dense_124/BiasAdd/ReadVariableOp ^dense_124/MatMul/ReadVariableOp3^dense_124/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_125/BiasAdd/ReadVariableOp ^dense_125/MatMul/ReadVariableOp3^dense_125/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_126/BiasAdd/ReadVariableOp ^dense_126/MatMul/ReadVariableOp3^dense_126/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_127/BiasAdd/ReadVariableOp ^dense_127/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������<Z: : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_86/batchnorm/ReadVariableOp/batch_normalization_86/batchnorm/ReadVariableOp2f
1batch_normalization_86/batchnorm/ReadVariableOp_11batch_normalization_86/batchnorm/ReadVariableOp_12f
1batch_normalization_86/batchnorm/ReadVariableOp_21batch_normalization_86/batchnorm/ReadVariableOp_22j
3batch_normalization_86/batchnorm/mul/ReadVariableOp3batch_normalization_86/batchnorm/mul/ReadVariableOp2b
/batch_normalization_87/batchnorm/ReadVariableOp/batch_normalization_87/batchnorm/ReadVariableOp2f
1batch_normalization_87/batchnorm/ReadVariableOp_11batch_normalization_87/batchnorm/ReadVariableOp_12f
1batch_normalization_87/batchnorm/ReadVariableOp_21batch_normalization_87/batchnorm/ReadVariableOp_22j
3batch_normalization_87/batchnorm/mul/ReadVariableOp3batch_normalization_87/batchnorm/mul/ReadVariableOp2b
/batch_normalization_88/batchnorm/ReadVariableOp/batch_normalization_88/batchnorm/ReadVariableOp2f
1batch_normalization_88/batchnorm/ReadVariableOp_11batch_normalization_88/batchnorm/ReadVariableOp_12f
1batch_normalization_88/batchnorm/ReadVariableOp_21batch_normalization_88/batchnorm/ReadVariableOp_22j
3batch_normalization_88/batchnorm/mul/ReadVariableOp3batch_normalization_88/batchnorm/mul/ReadVariableOp2D
 dense_124/BiasAdd/ReadVariableOp dense_124/BiasAdd/ReadVariableOp2B
dense_124/MatMul/ReadVariableOpdense_124/MatMul/ReadVariableOp2h
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_125/BiasAdd/ReadVariableOp dense_125/BiasAdd/ReadVariableOp2B
dense_125/MatMul/ReadVariableOpdense_125/MatMul/ReadVariableOp2h
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_126/BiasAdd/ReadVariableOp dense_126/BiasAdd/ReadVariableOp2B
dense_126/MatMul/ReadVariableOpdense_126/MatMul/ReadVariableOp2h
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_127/BiasAdd/ReadVariableOp dense_127/BiasAdd/ReadVariableOp2B
dense_127/MatMul/ReadVariableOpdense_127/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������<Z
 
_user_specified_nameinputs
߄
�
+__inference_model_16_layer_call_fn_14104310

inputs;
(dense_124_matmul_readvariableop_resource:	�*7
)dense_124_biasadd_readvariableop_resource:F
8batch_normalization_86_batchnorm_readvariableop_resource:J
<batch_normalization_86_batchnorm_mul_readvariableop_resource:H
:batch_normalization_86_batchnorm_readvariableop_1_resource:H
:batch_normalization_86_batchnorm_readvariableop_2_resource::
(dense_125_matmul_readvariableop_resource:@7
)dense_125_biasadd_readvariableop_resource:@F
8batch_normalization_87_batchnorm_readvariableop_resource:@J
<batch_normalization_87_batchnorm_mul_readvariableop_resource:@H
:batch_normalization_87_batchnorm_readvariableop_1_resource:@H
:batch_normalization_87_batchnorm_readvariableop_2_resource:@:
(dense_126_matmul_readvariableop_resource:@@7
)dense_126_biasadd_readvariableop_resource:@F
8batch_normalization_88_batchnorm_readvariableop_resource:@J
<batch_normalization_88_batchnorm_mul_readvariableop_resource:@H
:batch_normalization_88_batchnorm_readvariableop_1_resource:@H
:batch_normalization_88_batchnorm_readvariableop_2_resource:@:
(dense_127_matmul_readvariableop_resource:@7
)dense_127_biasadd_readvariableop_resource:
identity��/batch_normalization_86/batchnorm/ReadVariableOp�1batch_normalization_86/batchnorm/ReadVariableOp_1�1batch_normalization_86/batchnorm/ReadVariableOp_2�3batch_normalization_86/batchnorm/mul/ReadVariableOp�/batch_normalization_87/batchnorm/ReadVariableOp�1batch_normalization_87/batchnorm/ReadVariableOp_1�1batch_normalization_87/batchnorm/ReadVariableOp_2�3batch_normalization_87/batchnorm/mul/ReadVariableOp�/batch_normalization_88/batchnorm/ReadVariableOp�1batch_normalization_88/batchnorm/ReadVariableOp_1�1batch_normalization_88/batchnorm/ReadVariableOp_2�3batch_normalization_88/batchnorm/mul/ReadVariableOp� dense_124/BiasAdd/ReadVariableOp�dense_124/MatMul/ReadVariableOp�2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp� dense_125/BiasAdd/ReadVariableOp�dense_125/MatMul/ReadVariableOp�2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp� dense_126/BiasAdd/ReadVariableOp�dense_126/MatMul/ReadVariableOp�2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp� dense_127/BiasAdd/ReadVariableOp�dense_127/MatMul/ReadVariableOpa
flatten_39/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  s
flatten_39/ReshapeReshapeinputsflatten_39/Const:output:0*
T0*(
_output_shapes
:����������*�
dense_124/MatMul/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
dense_124/MatMulMatMulflatten_39/Reshape:output:0'dense_124/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_124/BiasAdd/ReadVariableOpReadVariableOp)dense_124_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_124/BiasAddBiasAdddense_124/MatMul:product:0(dense_124/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/batch_normalization_86/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_86_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_86/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_86/batchnorm/addAddV27batch_normalization_86/batchnorm/ReadVariableOp:value:0/batch_normalization_86/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_86/batchnorm/RsqrtRsqrt(batch_normalization_86/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_86/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_86_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_86/batchnorm/mulMul*batch_normalization_86/batchnorm/Rsqrt:y:0;batch_normalization_86/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_86/batchnorm/mul_1Muldense_124/BiasAdd:output:0(batch_normalization_86/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
1batch_normalization_86/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_86_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
&batch_normalization_86/batchnorm/mul_2Mul9batch_normalization_86/batchnorm/ReadVariableOp_1:value:0(batch_normalization_86/batchnorm/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_86/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_86_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
$batch_normalization_86/batchnorm/subSub9batch_normalization_86/batchnorm/ReadVariableOp_2:value:0*batch_normalization_86/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_86/batchnorm/add_1AddV2*batch_normalization_86/batchnorm/mul_1:z:0(batch_normalization_86/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������s
re_lu_44/ReluRelu*batch_normalization_86/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������n
dropout_10/IdentityIdentityre_lu_44/Relu:activations:0*
T0*'
_output_shapes
:����������
dense_125/MatMul/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_125/MatMulMatMuldropout_10/Identity:output:0'dense_125/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_125/BiasAdd/ReadVariableOpReadVariableOp)dense_125_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_125/BiasAddBiasAdddense_125/MatMul:product:0(dense_125/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/batch_normalization_87/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_87_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0k
&batch_normalization_87/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_87/batchnorm/addAddV27batch_normalization_87/batchnorm/ReadVariableOp:value:0/batch_normalization_87/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_87/batchnorm/RsqrtRsqrt(batch_normalization_87/batchnorm/add:z:0*
T0*
_output_shapes
:@�
3batch_normalization_87/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_87_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_87/batchnorm/mulMul*batch_normalization_87/batchnorm/Rsqrt:y:0;batch_normalization_87/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_87/batchnorm/mul_1Muldense_125/BiasAdd:output:0(batch_normalization_87/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
1batch_normalization_87/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_87_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_87/batchnorm/mul_2Mul9batch_normalization_87/batchnorm/ReadVariableOp_1:value:0(batch_normalization_87/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
1batch_normalization_87/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_87_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_87/batchnorm/subSub9batch_normalization_87/batchnorm/ReadVariableOp_2:value:0*batch_normalization_87/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_87/batchnorm/add_1AddV2*batch_normalization_87/batchnorm/mul_1:z:0(batch_normalization_87/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@s
re_lu_45/ReluRelu*batch_normalization_87/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@p
dropout_10/Identity_1Identityre_lu_45/Relu:activations:0*
T0*'
_output_shapes
:���������@�
dense_126/MatMul/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_126/MatMulMatMuldropout_10/Identity_1:output:0'dense_126/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_126/BiasAdd/ReadVariableOpReadVariableOp)dense_126_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_126/BiasAddBiasAdddense_126/MatMul:product:0(dense_126/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/batch_normalization_88/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_88_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0k
&batch_normalization_88/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_88/batchnorm/addAddV27batch_normalization_88/batchnorm/ReadVariableOp:value:0/batch_normalization_88/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_88/batchnorm/RsqrtRsqrt(batch_normalization_88/batchnorm/add:z:0*
T0*
_output_shapes
:@�
3batch_normalization_88/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_88_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_88/batchnorm/mulMul*batch_normalization_88/batchnorm/Rsqrt:y:0;batch_normalization_88/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_88/batchnorm/mul_1Muldense_126/BiasAdd:output:0(batch_normalization_88/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
1batch_normalization_88/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_88_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_88/batchnorm/mul_2Mul9batch_normalization_88/batchnorm/ReadVariableOp_1:value:0(batch_normalization_88/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
1batch_normalization_88/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_88_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_88/batchnorm/subSub9batch_normalization_88/batchnorm/ReadVariableOp_2:value:0*batch_normalization_88/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_88/batchnorm/add_1AddV2*batch_normalization_88/batchnorm/mul_1:z:0(batch_normalization_88/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@s
re_lu_46/ReluRelu*batch_normalization_88/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@�
dense_127/MatMul/ReadVariableOpReadVariableOp(dense_127_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_127/MatMulMatMulre_lu_46/Relu:activations:0'dense_127/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_127/BiasAdd/ReadVariableOpReadVariableOp)dense_127_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_127/BiasAddBiasAdddense_127/MatMul:product:0(dense_127/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
#dense_124/kernel/Regularizer/L2LossL2Loss:dense_124/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_124/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_124/kernel/Regularizer/mulMul+dense_124/kernel/Regularizer/mul/x:output:0,dense_124/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
#dense_125/kernel/Regularizer/L2LossL2Loss:dense_125/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_125/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_125/kernel/Regularizer/mulMul+dense_125/kernel/Regularizer/mul/x:output:0,dense_125/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
#dense_126/kernel/Regularizer/L2LossL2Loss:dense_126/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_126/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_126/kernel/Regularizer/mulMul+dense_126/kernel/Regularizer/mul/x:output:0,dense_126/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_127/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp0^batch_normalization_86/batchnorm/ReadVariableOp2^batch_normalization_86/batchnorm/ReadVariableOp_12^batch_normalization_86/batchnorm/ReadVariableOp_24^batch_normalization_86/batchnorm/mul/ReadVariableOp0^batch_normalization_87/batchnorm/ReadVariableOp2^batch_normalization_87/batchnorm/ReadVariableOp_12^batch_normalization_87/batchnorm/ReadVariableOp_24^batch_normalization_87/batchnorm/mul/ReadVariableOp0^batch_normalization_88/batchnorm/ReadVariableOp2^batch_normalization_88/batchnorm/ReadVariableOp_12^batch_normalization_88/batchnorm/ReadVariableOp_24^batch_normalization_88/batchnorm/mul/ReadVariableOp!^dense_124/BiasAdd/ReadVariableOp ^dense_124/MatMul/ReadVariableOp3^dense_124/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_125/BiasAdd/ReadVariableOp ^dense_125/MatMul/ReadVariableOp3^dense_125/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_126/BiasAdd/ReadVariableOp ^dense_126/MatMul/ReadVariableOp3^dense_126/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_127/BiasAdd/ReadVariableOp ^dense_127/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������<Z: : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_86/batchnorm/ReadVariableOp/batch_normalization_86/batchnorm/ReadVariableOp2f
1batch_normalization_86/batchnorm/ReadVariableOp_11batch_normalization_86/batchnorm/ReadVariableOp_12f
1batch_normalization_86/batchnorm/ReadVariableOp_21batch_normalization_86/batchnorm/ReadVariableOp_22j
3batch_normalization_86/batchnorm/mul/ReadVariableOp3batch_normalization_86/batchnorm/mul/ReadVariableOp2b
/batch_normalization_87/batchnorm/ReadVariableOp/batch_normalization_87/batchnorm/ReadVariableOp2f
1batch_normalization_87/batchnorm/ReadVariableOp_11batch_normalization_87/batchnorm/ReadVariableOp_12f
1batch_normalization_87/batchnorm/ReadVariableOp_21batch_normalization_87/batchnorm/ReadVariableOp_22j
3batch_normalization_87/batchnorm/mul/ReadVariableOp3batch_normalization_87/batchnorm/mul/ReadVariableOp2b
/batch_normalization_88/batchnorm/ReadVariableOp/batch_normalization_88/batchnorm/ReadVariableOp2f
1batch_normalization_88/batchnorm/ReadVariableOp_11batch_normalization_88/batchnorm/ReadVariableOp_12f
1batch_normalization_88/batchnorm/ReadVariableOp_21batch_normalization_88/batchnorm/ReadVariableOp_22j
3batch_normalization_88/batchnorm/mul/ReadVariableOp3batch_normalization_88/batchnorm/mul/ReadVariableOp2D
 dense_124/BiasAdd/ReadVariableOp dense_124/BiasAdd/ReadVariableOp2B
dense_124/MatMul/ReadVariableOpdense_124/MatMul/ReadVariableOp2h
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_125/BiasAdd/ReadVariableOp dense_125/BiasAdd/ReadVariableOp2B
dense_125/MatMul/ReadVariableOpdense_125/MatMul/ReadVariableOp2h
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_126/BiasAdd/ReadVariableOp dense_126/BiasAdd/ReadVariableOp2B
dense_126/MatMul/ReadVariableOpdense_126/MatMul/ReadVariableOp2h
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_127/BiasAdd/ReadVariableOp dense_127/BiasAdd/ReadVariableOp2B
dense_127/MatMul/ReadVariableOpdense_127/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������<Z
 
_user_specified_nameinputs
�$
�
9__inference_batch_normalization_88_layer_call_fn_14105161

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_86_layer_call_fn_14104767

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_model_16_layer_call_and_return_conditional_losses_14103977	
image;
(dense_124_matmul_readvariableop_resource:	�*7
)dense_124_biasadd_readvariableop_resource:F
8batch_normalization_86_batchnorm_readvariableop_resource:J
<batch_normalization_86_batchnorm_mul_readvariableop_resource:H
:batch_normalization_86_batchnorm_readvariableop_1_resource:H
:batch_normalization_86_batchnorm_readvariableop_2_resource::
(dense_125_matmul_readvariableop_resource:@7
)dense_125_biasadd_readvariableop_resource:@F
8batch_normalization_87_batchnorm_readvariableop_resource:@J
<batch_normalization_87_batchnorm_mul_readvariableop_resource:@H
:batch_normalization_87_batchnorm_readvariableop_1_resource:@H
:batch_normalization_87_batchnorm_readvariableop_2_resource:@:
(dense_126_matmul_readvariableop_resource:@@7
)dense_126_biasadd_readvariableop_resource:@F
8batch_normalization_88_batchnorm_readvariableop_resource:@J
<batch_normalization_88_batchnorm_mul_readvariableop_resource:@H
:batch_normalization_88_batchnorm_readvariableop_1_resource:@H
:batch_normalization_88_batchnorm_readvariableop_2_resource:@:
(dense_127_matmul_readvariableop_resource:@7
)dense_127_biasadd_readvariableop_resource:
identity��/batch_normalization_86/batchnorm/ReadVariableOp�1batch_normalization_86/batchnorm/ReadVariableOp_1�1batch_normalization_86/batchnorm/ReadVariableOp_2�3batch_normalization_86/batchnorm/mul/ReadVariableOp�/batch_normalization_87/batchnorm/ReadVariableOp�1batch_normalization_87/batchnorm/ReadVariableOp_1�1batch_normalization_87/batchnorm/ReadVariableOp_2�3batch_normalization_87/batchnorm/mul/ReadVariableOp�/batch_normalization_88/batchnorm/ReadVariableOp�1batch_normalization_88/batchnorm/ReadVariableOp_1�1batch_normalization_88/batchnorm/ReadVariableOp_2�3batch_normalization_88/batchnorm/mul/ReadVariableOp� dense_124/BiasAdd/ReadVariableOp�dense_124/MatMul/ReadVariableOp�<dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp�2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp� dense_125/BiasAdd/ReadVariableOp�dense_125/MatMul/ReadVariableOp�<dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp�2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp� dense_126/BiasAdd/ReadVariableOp�dense_126/MatMul/ReadVariableOp�<dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp�2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp� dense_127/BiasAdd/ReadVariableOp�dense_127/MatMul/ReadVariableOpa
flatten_39/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  r
flatten_39/ReshapeReshapeimageflatten_39/Const:output:0*
T0*(
_output_shapes
:����������*�
dense_124/MatMul/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
dense_124/MatMulMatMulflatten_39/Reshape:output:0'dense_124/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_124/BiasAdd/ReadVariableOpReadVariableOp)dense_124_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_124/BiasAddBiasAdddense_124/MatMul:product:0(dense_124/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
<dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
-dense_124/dense_124/kernel/Regularizer/L2LossL2LossDdense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: q
,dense_124/dense_124/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
*dense_124/dense_124/kernel/Regularizer/mulMul5dense_124/dense_124/kernel/Regularizer/mul/x:output:06dense_124/dense_124/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/batch_normalization_86/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_86_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_86/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_86/batchnorm/addAddV27batch_normalization_86/batchnorm/ReadVariableOp:value:0/batch_normalization_86/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_86/batchnorm/RsqrtRsqrt(batch_normalization_86/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_86/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_86_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_86/batchnorm/mulMul*batch_normalization_86/batchnorm/Rsqrt:y:0;batch_normalization_86/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_86/batchnorm/mul_1Muldense_124/BiasAdd:output:0(batch_normalization_86/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
1batch_normalization_86/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_86_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
&batch_normalization_86/batchnorm/mul_2Mul9batch_normalization_86/batchnorm/ReadVariableOp_1:value:0(batch_normalization_86/batchnorm/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_86/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_86_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
$batch_normalization_86/batchnorm/subSub9batch_normalization_86/batchnorm/ReadVariableOp_2:value:0*batch_normalization_86/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_86/batchnorm/add_1AddV2*batch_normalization_86/batchnorm/mul_1:z:0(batch_normalization_86/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������s
re_lu_44/ReluRelu*batch_normalization_86/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������n
dropout_10/IdentityIdentityre_lu_44/Relu:activations:0*
T0*'
_output_shapes
:����������
dense_125/MatMul/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_125/MatMulMatMuldropout_10/Identity:output:0'dense_125/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_125/BiasAdd/ReadVariableOpReadVariableOp)dense_125_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_125/BiasAddBiasAdddense_125/MatMul:product:0(dense_125/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
-dense_125/dense_125/kernel/Regularizer/L2LossL2LossDdense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: q
,dense_125/dense_125/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
*dense_125/dense_125/kernel/Regularizer/mulMul5dense_125/dense_125/kernel/Regularizer/mul/x:output:06dense_125/dense_125/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/batch_normalization_87/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_87_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0k
&batch_normalization_87/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_87/batchnorm/addAddV27batch_normalization_87/batchnorm/ReadVariableOp:value:0/batch_normalization_87/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_87/batchnorm/RsqrtRsqrt(batch_normalization_87/batchnorm/add:z:0*
T0*
_output_shapes
:@�
3batch_normalization_87/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_87_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_87/batchnorm/mulMul*batch_normalization_87/batchnorm/Rsqrt:y:0;batch_normalization_87/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_87/batchnorm/mul_1Muldense_125/BiasAdd:output:0(batch_normalization_87/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
1batch_normalization_87/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_87_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_87/batchnorm/mul_2Mul9batch_normalization_87/batchnorm/ReadVariableOp_1:value:0(batch_normalization_87/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
1batch_normalization_87/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_87_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_87/batchnorm/subSub9batch_normalization_87/batchnorm/ReadVariableOp_2:value:0*batch_normalization_87/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_87/batchnorm/add_1AddV2*batch_normalization_87/batchnorm/mul_1:z:0(batch_normalization_87/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@s
re_lu_45/ReluRelu*batch_normalization_87/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@p
dropout_10/Identity_1Identityre_lu_45/Relu:activations:0*
T0*'
_output_shapes
:���������@�
dense_126/MatMul/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_126/MatMulMatMuldropout_10/Identity_1:output:0'dense_126/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_126/BiasAdd/ReadVariableOpReadVariableOp)dense_126_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_126/BiasAddBiasAdddense_126/MatMul:product:0(dense_126/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
-dense_126/dense_126/kernel/Regularizer/L2LossL2LossDdense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: q
,dense_126/dense_126/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
*dense_126/dense_126/kernel/Regularizer/mulMul5dense_126/dense_126/kernel/Regularizer/mul/x:output:06dense_126/dense_126/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/batch_normalization_88/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_88_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0k
&batch_normalization_88/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_88/batchnorm/addAddV27batch_normalization_88/batchnorm/ReadVariableOp:value:0/batch_normalization_88/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_88/batchnorm/RsqrtRsqrt(batch_normalization_88/batchnorm/add:z:0*
T0*
_output_shapes
:@�
3batch_normalization_88/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_88_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_88/batchnorm/mulMul*batch_normalization_88/batchnorm/Rsqrt:y:0;batch_normalization_88/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_88/batchnorm/mul_1Muldense_126/BiasAdd:output:0(batch_normalization_88/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
1batch_normalization_88/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_88_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_88/batchnorm/mul_2Mul9batch_normalization_88/batchnorm/ReadVariableOp_1:value:0(batch_normalization_88/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
1batch_normalization_88/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_88_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_88/batchnorm/subSub9batch_normalization_88/batchnorm/ReadVariableOp_2:value:0*batch_normalization_88/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_88/batchnorm/add_1AddV2*batch_normalization_88/batchnorm/mul_1:z:0(batch_normalization_88/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@s
re_lu_46/ReluRelu*batch_normalization_88/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@�
dense_127/MatMul/ReadVariableOpReadVariableOp(dense_127_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_127/MatMulMatMulre_lu_46/Relu:activations:0'dense_127/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_127/BiasAdd/ReadVariableOpReadVariableOp)dense_127_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_127/BiasAddBiasAdddense_127/MatMul:product:0(dense_127/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
#dense_124/kernel/Regularizer/L2LossL2Loss:dense_124/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_124/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_124/kernel/Regularizer/mulMul+dense_124/kernel/Regularizer/mul/x:output:0,dense_124/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
#dense_125/kernel/Regularizer/L2LossL2Loss:dense_125/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_125/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_125/kernel/Regularizer/mulMul+dense_125/kernel/Regularizer/mul/x:output:0,dense_125/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
#dense_126/kernel/Regularizer/L2LossL2Loss:dense_126/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_126/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_126/kernel/Regularizer/mulMul+dense_126/kernel/Regularizer/mul/x:output:0,dense_126/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_127/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������

NoOpNoOp0^batch_normalization_86/batchnorm/ReadVariableOp2^batch_normalization_86/batchnorm/ReadVariableOp_12^batch_normalization_86/batchnorm/ReadVariableOp_24^batch_normalization_86/batchnorm/mul/ReadVariableOp0^batch_normalization_87/batchnorm/ReadVariableOp2^batch_normalization_87/batchnorm/ReadVariableOp_12^batch_normalization_87/batchnorm/ReadVariableOp_24^batch_normalization_87/batchnorm/mul/ReadVariableOp0^batch_normalization_88/batchnorm/ReadVariableOp2^batch_normalization_88/batchnorm/ReadVariableOp_12^batch_normalization_88/batchnorm/ReadVariableOp_24^batch_normalization_88/batchnorm/mul/ReadVariableOp!^dense_124/BiasAdd/ReadVariableOp ^dense_124/MatMul/ReadVariableOp=^dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp3^dense_124/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_125/BiasAdd/ReadVariableOp ^dense_125/MatMul/ReadVariableOp=^dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp3^dense_125/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_126/BiasAdd/ReadVariableOp ^dense_126/MatMul/ReadVariableOp=^dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp3^dense_126/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_127/BiasAdd/ReadVariableOp ^dense_127/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������<Z: : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_86/batchnorm/ReadVariableOp/batch_normalization_86/batchnorm/ReadVariableOp2f
1batch_normalization_86/batchnorm/ReadVariableOp_11batch_normalization_86/batchnorm/ReadVariableOp_12f
1batch_normalization_86/batchnorm/ReadVariableOp_21batch_normalization_86/batchnorm/ReadVariableOp_22j
3batch_normalization_86/batchnorm/mul/ReadVariableOp3batch_normalization_86/batchnorm/mul/ReadVariableOp2b
/batch_normalization_87/batchnorm/ReadVariableOp/batch_normalization_87/batchnorm/ReadVariableOp2f
1batch_normalization_87/batchnorm/ReadVariableOp_11batch_normalization_87/batchnorm/ReadVariableOp_12f
1batch_normalization_87/batchnorm/ReadVariableOp_21batch_normalization_87/batchnorm/ReadVariableOp_22j
3batch_normalization_87/batchnorm/mul/ReadVariableOp3batch_normalization_87/batchnorm/mul/ReadVariableOp2b
/batch_normalization_88/batchnorm/ReadVariableOp/batch_normalization_88/batchnorm/ReadVariableOp2f
1batch_normalization_88/batchnorm/ReadVariableOp_11batch_normalization_88/batchnorm/ReadVariableOp_12f
1batch_normalization_88/batchnorm/ReadVariableOp_21batch_normalization_88/batchnorm/ReadVariableOp_22j
3batch_normalization_88/batchnorm/mul/ReadVariableOp3batch_normalization_88/batchnorm/mul/ReadVariableOp2D
 dense_124/BiasAdd/ReadVariableOp dense_124/BiasAdd/ReadVariableOp2B
dense_124/MatMul/ReadVariableOpdense_124/MatMul/ReadVariableOp2|
<dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp<dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2h
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_125/BiasAdd/ReadVariableOp dense_125/BiasAdd/ReadVariableOp2B
dense_125/MatMul/ReadVariableOpdense_125/MatMul/ReadVariableOp2|
<dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp<dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2h
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_126/BiasAdd/ReadVariableOp dense_126/BiasAdd/ReadVariableOp2B
dense_126/MatMul/ReadVariableOpdense_126/MatMul/ReadVariableOp2|
<dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp<dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2h
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_127/BiasAdd/ReadVariableOp dense_127/BiasAdd/ReadVariableOp2B
dense_127/MatMul/ReadVariableOpdense_127/MatMul/ReadVariableOp:V R
/
_output_shapes
:���������<Z

_user_specified_nameimage
�
G
+__inference_re_lu_46_layer_call_fn_14105220

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:���������@Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_0_14105254N
;dense_124_kernel_regularizer_l2loss_readvariableop_resource:	�*
identity��2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp�
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp;dense_124_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	�**
dtype0�
#dense_124/kernel/Regularizer/L2LossL2Loss:dense_124/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_124/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_124/kernel/Regularizer/mulMul+dense_124/kernel/Regularizer/mul/x:output:0,dense_124/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_124/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^dense_124/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp
��
�
F__inference_model_16_layer_call_and_return_conditional_losses_14104140	
image;
(dense_124_matmul_readvariableop_resource:	�*7
)dense_124_biasadd_readvariableop_resource:L
>batch_normalization_86_assignmovingavg_readvariableop_resource:N
@batch_normalization_86_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_86_batchnorm_mul_readvariableop_resource:F
8batch_normalization_86_batchnorm_readvariableop_resource::
(dense_125_matmul_readvariableop_resource:@7
)dense_125_biasadd_readvariableop_resource:@L
>batch_normalization_87_assignmovingavg_readvariableop_resource:@N
@batch_normalization_87_assignmovingavg_1_readvariableop_resource:@J
<batch_normalization_87_batchnorm_mul_readvariableop_resource:@F
8batch_normalization_87_batchnorm_readvariableop_resource:@:
(dense_126_matmul_readvariableop_resource:@@7
)dense_126_biasadd_readvariableop_resource:@L
>batch_normalization_88_assignmovingavg_readvariableop_resource:@N
@batch_normalization_88_assignmovingavg_1_readvariableop_resource:@J
<batch_normalization_88_batchnorm_mul_readvariableop_resource:@F
8batch_normalization_88_batchnorm_readvariableop_resource:@:
(dense_127_matmul_readvariableop_resource:@7
)dense_127_biasadd_readvariableop_resource:
identity��&batch_normalization_86/AssignMovingAvg�5batch_normalization_86/AssignMovingAvg/ReadVariableOp�(batch_normalization_86/AssignMovingAvg_1�7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_86/batchnorm/ReadVariableOp�3batch_normalization_86/batchnorm/mul/ReadVariableOp�&batch_normalization_87/AssignMovingAvg�5batch_normalization_87/AssignMovingAvg/ReadVariableOp�(batch_normalization_87/AssignMovingAvg_1�7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_87/batchnorm/ReadVariableOp�3batch_normalization_87/batchnorm/mul/ReadVariableOp�&batch_normalization_88/AssignMovingAvg�5batch_normalization_88/AssignMovingAvg/ReadVariableOp�(batch_normalization_88/AssignMovingAvg_1�7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_88/batchnorm/ReadVariableOp�3batch_normalization_88/batchnorm/mul/ReadVariableOp� dense_124/BiasAdd/ReadVariableOp�dense_124/MatMul/ReadVariableOp�<dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp�2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp� dense_125/BiasAdd/ReadVariableOp�dense_125/MatMul/ReadVariableOp�<dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp�2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp� dense_126/BiasAdd/ReadVariableOp�dense_126/MatMul/ReadVariableOp�<dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp�2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp� dense_127/BiasAdd/ReadVariableOp�dense_127/MatMul/ReadVariableOpa
flatten_39/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  r
flatten_39/ReshapeReshapeimageflatten_39/Const:output:0*
T0*(
_output_shapes
:����������*�
dense_124/MatMul/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
dense_124/MatMulMatMulflatten_39/Reshape:output:0'dense_124/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_124/BiasAdd/ReadVariableOpReadVariableOp)dense_124_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_124/BiasAddBiasAdddense_124/MatMul:product:0(dense_124/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
<dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
-dense_124/dense_124/kernel/Regularizer/L2LossL2LossDdense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: q
,dense_124/dense_124/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
*dense_124/dense_124/kernel/Regularizer/mulMul5dense_124/dense_124/kernel/Regularizer/mul/x:output:06dense_124/dense_124/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5batch_normalization_86/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_86/moments/meanMeandense_124/BiasAdd:output:0>batch_normalization_86/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
+batch_normalization_86/moments/StopGradientStopGradient,batch_normalization_86/moments/mean:output:0*
T0*
_output_shapes

:�
0batch_normalization_86/moments/SquaredDifferenceSquaredDifferencedense_124/BiasAdd:output:04batch_normalization_86/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
9batch_normalization_86/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_86/moments/varianceMean4batch_normalization_86/moments/SquaredDifference:z:0Bbatch_normalization_86/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
&batch_normalization_86/moments/SqueezeSqueeze,batch_normalization_86/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
(batch_normalization_86/moments/Squeeze_1Squeeze0batch_normalization_86/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_86/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_86/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_86_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_86/AssignMovingAvg/subSub=batch_normalization_86/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_86/moments/Squeeze:output:0*
T0*
_output_shapes
:�
*batch_normalization_86/AssignMovingAvg/mulMul.batch_normalization_86/AssignMovingAvg/sub:z:05batch_normalization_86/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
&batch_normalization_86/AssignMovingAvgAssignSubVariableOp>batch_normalization_86_assignmovingavg_readvariableop_resource.batch_normalization_86/AssignMovingAvg/mul:z:06^batch_normalization_86/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_86/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_86/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_86_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_86/AssignMovingAvg_1/subSub?batch_normalization_86/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_86/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
,batch_normalization_86/AssignMovingAvg_1/mulMul0batch_normalization_86/AssignMovingAvg_1/sub:z:07batch_normalization_86/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_86/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_86_assignmovingavg_1_readvariableop_resource0batch_normalization_86/AssignMovingAvg_1/mul:z:08^batch_normalization_86/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_86/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_86/batchnorm/addAddV21batch_normalization_86/moments/Squeeze_1:output:0/batch_normalization_86/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_86/batchnorm/RsqrtRsqrt(batch_normalization_86/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_86/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_86_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_86/batchnorm/mulMul*batch_normalization_86/batchnorm/Rsqrt:y:0;batch_normalization_86/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_86/batchnorm/mul_1Muldense_124/BiasAdd:output:0(batch_normalization_86/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
&batch_normalization_86/batchnorm/mul_2Mul/batch_normalization_86/moments/Squeeze:output:0(batch_normalization_86/batchnorm/mul:z:0*
T0*
_output_shapes
:�
/batch_normalization_86/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_86_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_86/batchnorm/subSub7batch_normalization_86/batchnorm/ReadVariableOp:value:0*batch_normalization_86/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_86/batchnorm/add_1AddV2*batch_normalization_86/batchnorm/mul_1:z:0(batch_normalization_86/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������s
re_lu_44/ReluRelu*batch_normalization_86/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������]
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_10/dropout/MulMulre_lu_44/Relu:activations:0!dropout_10/dropout/Const:output:0*
T0*'
_output_shapes
:���������c
dropout_10/dropout/ShapeShapere_lu_44/Relu:activations:0*
T0*
_output_shapes
:�
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0f
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*'
_output_shapes
:����������
dense_125/MatMul/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_125/MatMulMatMuldropout_10/dropout/Mul_1:z:0'dense_125/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_125/BiasAdd/ReadVariableOpReadVariableOp)dense_125_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_125/BiasAddBiasAdddense_125/MatMul:product:0(dense_125/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
-dense_125/dense_125/kernel/Regularizer/L2LossL2LossDdense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: q
,dense_125/dense_125/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
*dense_125/dense_125/kernel/Regularizer/mulMul5dense_125/dense_125/kernel/Regularizer/mul/x:output:06dense_125/dense_125/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5batch_normalization_87/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_87/moments/meanMeandense_125/BiasAdd:output:0>batch_normalization_87/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
+batch_normalization_87/moments/StopGradientStopGradient,batch_normalization_87/moments/mean:output:0*
T0*
_output_shapes

:@�
0batch_normalization_87/moments/SquaredDifferenceSquaredDifferencedense_125/BiasAdd:output:04batch_normalization_87/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������@�
9batch_normalization_87/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_87/moments/varianceMean4batch_normalization_87/moments/SquaredDifference:z:0Bbatch_normalization_87/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
&batch_normalization_87/moments/SqueezeSqueeze,batch_normalization_87/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 �
(batch_normalization_87/moments/Squeeze_1Squeeze0batch_normalization_87/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 q
,batch_normalization_87/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_87/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_87_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
*batch_normalization_87/AssignMovingAvg/subSub=batch_normalization_87/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_87/moments/Squeeze:output:0*
T0*
_output_shapes
:@�
*batch_normalization_87/AssignMovingAvg/mulMul.batch_normalization_87/AssignMovingAvg/sub:z:05batch_normalization_87/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
&batch_normalization_87/AssignMovingAvgAssignSubVariableOp>batch_normalization_87_assignmovingavg_readvariableop_resource.batch_normalization_87/AssignMovingAvg/mul:z:06^batch_normalization_87/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_87/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_87/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_87_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
,batch_normalization_87/AssignMovingAvg_1/subSub?batch_normalization_87/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_87/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@�
,batch_normalization_87/AssignMovingAvg_1/mulMul0batch_normalization_87/AssignMovingAvg_1/sub:z:07batch_normalization_87/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
(batch_normalization_87/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_87_assignmovingavg_1_readvariableop_resource0batch_normalization_87/AssignMovingAvg_1/mul:z:08^batch_normalization_87/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_87/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_87/batchnorm/addAddV21batch_normalization_87/moments/Squeeze_1:output:0/batch_normalization_87/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_87/batchnorm/RsqrtRsqrt(batch_normalization_87/batchnorm/add:z:0*
T0*
_output_shapes
:@�
3batch_normalization_87/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_87_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_87/batchnorm/mulMul*batch_normalization_87/batchnorm/Rsqrt:y:0;batch_normalization_87/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_87/batchnorm/mul_1Muldense_125/BiasAdd:output:0(batch_normalization_87/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
&batch_normalization_87/batchnorm/mul_2Mul/batch_normalization_87/moments/Squeeze:output:0(batch_normalization_87/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
/batch_normalization_87/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_87_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_87/batchnorm/subSub7batch_normalization_87/batchnorm/ReadVariableOp:value:0*batch_normalization_87/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_87/batchnorm/add_1AddV2*batch_normalization_87/batchnorm/mul_1:z:0(batch_normalization_87/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@s
re_lu_45/ReluRelu*batch_normalization_87/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@_
dropout_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_10/dropout_1/MulMulre_lu_45/Relu:activations:0#dropout_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:���������@e
dropout_10/dropout_1/ShapeShapere_lu_45/Relu:activations:0*
T0*
_output_shapes
:�
1dropout_10/dropout_1/random_uniform/RandomUniformRandomUniform#dropout_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0h
#dropout_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
!dropout_10/dropout_1/GreaterEqualGreaterEqual:dropout_10/dropout_1/random_uniform/RandomUniform:output:0,dropout_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
dropout_10/dropout_1/CastCast%dropout_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
dropout_10/dropout_1/Mul_1Muldropout_10/dropout_1/Mul:z:0dropout_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:���������@�
dense_126/MatMul/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_126/MatMulMatMuldropout_10/dropout_1/Mul_1:z:0'dense_126/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_126/BiasAdd/ReadVariableOpReadVariableOp)dense_126_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_126/BiasAddBiasAdddense_126/MatMul:product:0(dense_126/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
<dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
-dense_126/dense_126/kernel/Regularizer/L2LossL2LossDdense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: q
,dense_126/dense_126/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
*dense_126/dense_126/kernel/Regularizer/mulMul5dense_126/dense_126/kernel/Regularizer/mul/x:output:06dense_126/dense_126/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5batch_normalization_88/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_88/moments/meanMeandense_126/BiasAdd:output:0>batch_normalization_88/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
+batch_normalization_88/moments/StopGradientStopGradient,batch_normalization_88/moments/mean:output:0*
T0*
_output_shapes

:@�
0batch_normalization_88/moments/SquaredDifferenceSquaredDifferencedense_126/BiasAdd:output:04batch_normalization_88/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������@�
9batch_normalization_88/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_88/moments/varianceMean4batch_normalization_88/moments/SquaredDifference:z:0Bbatch_normalization_88/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
&batch_normalization_88/moments/SqueezeSqueeze,batch_normalization_88/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 �
(batch_normalization_88/moments/Squeeze_1Squeeze0batch_normalization_88/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 q
,batch_normalization_88/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_88/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_88_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
*batch_normalization_88/AssignMovingAvg/subSub=batch_normalization_88/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_88/moments/Squeeze:output:0*
T0*
_output_shapes
:@�
*batch_normalization_88/AssignMovingAvg/mulMul.batch_normalization_88/AssignMovingAvg/sub:z:05batch_normalization_88/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
&batch_normalization_88/AssignMovingAvgAssignSubVariableOp>batch_normalization_88_assignmovingavg_readvariableop_resource.batch_normalization_88/AssignMovingAvg/mul:z:06^batch_normalization_88/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_88/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_88/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_88_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
,batch_normalization_88/AssignMovingAvg_1/subSub?batch_normalization_88/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_88/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@�
,batch_normalization_88/AssignMovingAvg_1/mulMul0batch_normalization_88/AssignMovingAvg_1/sub:z:07batch_normalization_88/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
(batch_normalization_88/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_88_assignmovingavg_1_readvariableop_resource0batch_normalization_88/AssignMovingAvg_1/mul:z:08^batch_normalization_88/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_88/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_88/batchnorm/addAddV21batch_normalization_88/moments/Squeeze_1:output:0/batch_normalization_88/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_88/batchnorm/RsqrtRsqrt(batch_normalization_88/batchnorm/add:z:0*
T0*
_output_shapes
:@�
3batch_normalization_88/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_88_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_88/batchnorm/mulMul*batch_normalization_88/batchnorm/Rsqrt:y:0;batch_normalization_88/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_88/batchnorm/mul_1Muldense_126/BiasAdd:output:0(batch_normalization_88/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
&batch_normalization_88/batchnorm/mul_2Mul/batch_normalization_88/moments/Squeeze:output:0(batch_normalization_88/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
/batch_normalization_88/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_88_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_88/batchnorm/subSub7batch_normalization_88/batchnorm/ReadVariableOp:value:0*batch_normalization_88/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_88/batchnorm/add_1AddV2*batch_normalization_88/batchnorm/mul_1:z:0(batch_normalization_88/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@s
re_lu_46/ReluRelu*batch_normalization_88/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@�
dense_127/MatMul/ReadVariableOpReadVariableOp(dense_127_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_127/MatMulMatMulre_lu_46/Relu:activations:0'dense_127/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_127/BiasAdd/ReadVariableOpReadVariableOp)dense_127_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_127/BiasAddBiasAdddense_127/MatMul:product:0(dense_127/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
#dense_124/kernel/Regularizer/L2LossL2Loss:dense_124/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_124/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_124/kernel/Regularizer/mulMul+dense_124/kernel/Regularizer/mul/x:output:0,dense_124/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
#dense_125/kernel/Regularizer/L2LossL2Loss:dense_125/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_125/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_125/kernel/Regularizer/mulMul+dense_125/kernel/Regularizer/mul/x:output:0,dense_125/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
#dense_126/kernel/Regularizer/L2LossL2Loss:dense_126/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_126/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_126/kernel/Regularizer/mulMul+dense_126/kernel/Regularizer/mul/x:output:0,dense_126/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_127/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^batch_normalization_86/AssignMovingAvg6^batch_normalization_86/AssignMovingAvg/ReadVariableOp)^batch_normalization_86/AssignMovingAvg_18^batch_normalization_86/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_86/batchnorm/ReadVariableOp4^batch_normalization_86/batchnorm/mul/ReadVariableOp'^batch_normalization_87/AssignMovingAvg6^batch_normalization_87/AssignMovingAvg/ReadVariableOp)^batch_normalization_87/AssignMovingAvg_18^batch_normalization_87/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_87/batchnorm/ReadVariableOp4^batch_normalization_87/batchnorm/mul/ReadVariableOp'^batch_normalization_88/AssignMovingAvg6^batch_normalization_88/AssignMovingAvg/ReadVariableOp)^batch_normalization_88/AssignMovingAvg_18^batch_normalization_88/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_88/batchnorm/ReadVariableOp4^batch_normalization_88/batchnorm/mul/ReadVariableOp!^dense_124/BiasAdd/ReadVariableOp ^dense_124/MatMul/ReadVariableOp=^dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp3^dense_124/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_125/BiasAdd/ReadVariableOp ^dense_125/MatMul/ReadVariableOp=^dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp3^dense_125/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_126/BiasAdd/ReadVariableOp ^dense_126/MatMul/ReadVariableOp=^dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp3^dense_126/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_127/BiasAdd/ReadVariableOp ^dense_127/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������<Z: : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_86/AssignMovingAvg&batch_normalization_86/AssignMovingAvg2n
5batch_normalization_86/AssignMovingAvg/ReadVariableOp5batch_normalization_86/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_86/AssignMovingAvg_1(batch_normalization_86/AssignMovingAvg_12r
7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_86/batchnorm/ReadVariableOp/batch_normalization_86/batchnorm/ReadVariableOp2j
3batch_normalization_86/batchnorm/mul/ReadVariableOp3batch_normalization_86/batchnorm/mul/ReadVariableOp2P
&batch_normalization_87/AssignMovingAvg&batch_normalization_87/AssignMovingAvg2n
5batch_normalization_87/AssignMovingAvg/ReadVariableOp5batch_normalization_87/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_87/AssignMovingAvg_1(batch_normalization_87/AssignMovingAvg_12r
7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_87/batchnorm/ReadVariableOp/batch_normalization_87/batchnorm/ReadVariableOp2j
3batch_normalization_87/batchnorm/mul/ReadVariableOp3batch_normalization_87/batchnorm/mul/ReadVariableOp2P
&batch_normalization_88/AssignMovingAvg&batch_normalization_88/AssignMovingAvg2n
5batch_normalization_88/AssignMovingAvg/ReadVariableOp5batch_normalization_88/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_88/AssignMovingAvg_1(batch_normalization_88/AssignMovingAvg_12r
7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_88/batchnorm/ReadVariableOp/batch_normalization_88/batchnorm/ReadVariableOp2j
3batch_normalization_88/batchnorm/mul/ReadVariableOp3batch_normalization_88/batchnorm/mul/ReadVariableOp2D
 dense_124/BiasAdd/ReadVariableOp dense_124/BiasAdd/ReadVariableOp2B
dense_124/MatMul/ReadVariableOpdense_124/MatMul/ReadVariableOp2|
<dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp<dense_124/dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2h
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_125/BiasAdd/ReadVariableOp dense_125/BiasAdd/ReadVariableOp2B
dense_125/MatMul/ReadVariableOpdense_125/MatMul/ReadVariableOp2|
<dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp<dense_125/dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2h
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_126/BiasAdd/ReadVariableOp dense_126/BiasAdd/ReadVariableOp2B
dense_126/MatMul/ReadVariableOpdense_126/MatMul/ReadVariableOp2|
<dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp<dense_126/dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2h
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_127/BiasAdd/ReadVariableOp dense_127/BiasAdd/ReadVariableOp2B
dense_127/MatMul/ReadVariableOpdense_127/MatMul/ReadVariableOp:V R
/
_output_shapes
:���������<Z

_user_specified_nameimage
�$
�
9__inference_batch_normalization_87_layer_call_fn_14105015

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�
F__inference_model_16_layer_call_and_return_conditional_losses_14104707

inputs;
(dense_124_matmul_readvariableop_resource:	�*7
)dense_124_biasadd_readvariableop_resource:L
>batch_normalization_86_assignmovingavg_readvariableop_resource:N
@batch_normalization_86_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_86_batchnorm_mul_readvariableop_resource:F
8batch_normalization_86_batchnorm_readvariableop_resource::
(dense_125_matmul_readvariableop_resource:@7
)dense_125_biasadd_readvariableop_resource:@L
>batch_normalization_87_assignmovingavg_readvariableop_resource:@N
@batch_normalization_87_assignmovingavg_1_readvariableop_resource:@J
<batch_normalization_87_batchnorm_mul_readvariableop_resource:@F
8batch_normalization_87_batchnorm_readvariableop_resource:@:
(dense_126_matmul_readvariableop_resource:@@7
)dense_126_biasadd_readvariableop_resource:@L
>batch_normalization_88_assignmovingavg_readvariableop_resource:@N
@batch_normalization_88_assignmovingavg_1_readvariableop_resource:@J
<batch_normalization_88_batchnorm_mul_readvariableop_resource:@F
8batch_normalization_88_batchnorm_readvariableop_resource:@:
(dense_127_matmul_readvariableop_resource:@7
)dense_127_biasadd_readvariableop_resource:
identity��&batch_normalization_86/AssignMovingAvg�5batch_normalization_86/AssignMovingAvg/ReadVariableOp�(batch_normalization_86/AssignMovingAvg_1�7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_86/batchnorm/ReadVariableOp�3batch_normalization_86/batchnorm/mul/ReadVariableOp�&batch_normalization_87/AssignMovingAvg�5batch_normalization_87/AssignMovingAvg/ReadVariableOp�(batch_normalization_87/AssignMovingAvg_1�7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_87/batchnorm/ReadVariableOp�3batch_normalization_87/batchnorm/mul/ReadVariableOp�&batch_normalization_88/AssignMovingAvg�5batch_normalization_88/AssignMovingAvg/ReadVariableOp�(batch_normalization_88/AssignMovingAvg_1�7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_88/batchnorm/ReadVariableOp�3batch_normalization_88/batchnorm/mul/ReadVariableOp� dense_124/BiasAdd/ReadVariableOp�dense_124/MatMul/ReadVariableOp�2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp� dense_125/BiasAdd/ReadVariableOp�dense_125/MatMul/ReadVariableOp�2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp� dense_126/BiasAdd/ReadVariableOp�dense_126/MatMul/ReadVariableOp�2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp� dense_127/BiasAdd/ReadVariableOp�dense_127/MatMul/ReadVariableOpa
flatten_39/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  s
flatten_39/ReshapeReshapeinputsflatten_39/Const:output:0*
T0*(
_output_shapes
:����������*�
dense_124/MatMul/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
dense_124/MatMulMatMulflatten_39/Reshape:output:0'dense_124/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_124/BiasAdd/ReadVariableOpReadVariableOp)dense_124_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_124/BiasAddBiasAdddense_124/MatMul:product:0(dense_124/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
5batch_normalization_86/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_86/moments/meanMeandense_124/BiasAdd:output:0>batch_normalization_86/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
+batch_normalization_86/moments/StopGradientStopGradient,batch_normalization_86/moments/mean:output:0*
T0*
_output_shapes

:�
0batch_normalization_86/moments/SquaredDifferenceSquaredDifferencedense_124/BiasAdd:output:04batch_normalization_86/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
9batch_normalization_86/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_86/moments/varianceMean4batch_normalization_86/moments/SquaredDifference:z:0Bbatch_normalization_86/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
&batch_normalization_86/moments/SqueezeSqueeze,batch_normalization_86/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
(batch_normalization_86/moments/Squeeze_1Squeeze0batch_normalization_86/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_86/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_86/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_86_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_86/AssignMovingAvg/subSub=batch_normalization_86/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_86/moments/Squeeze:output:0*
T0*
_output_shapes
:�
*batch_normalization_86/AssignMovingAvg/mulMul.batch_normalization_86/AssignMovingAvg/sub:z:05batch_normalization_86/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
&batch_normalization_86/AssignMovingAvgAssignSubVariableOp>batch_normalization_86_assignmovingavg_readvariableop_resource.batch_normalization_86/AssignMovingAvg/mul:z:06^batch_normalization_86/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_86/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_86/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_86_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_86/AssignMovingAvg_1/subSub?batch_normalization_86/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_86/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
,batch_normalization_86/AssignMovingAvg_1/mulMul0batch_normalization_86/AssignMovingAvg_1/sub:z:07batch_normalization_86/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_86/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_86_assignmovingavg_1_readvariableop_resource0batch_normalization_86/AssignMovingAvg_1/mul:z:08^batch_normalization_86/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_86/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_86/batchnorm/addAddV21batch_normalization_86/moments/Squeeze_1:output:0/batch_normalization_86/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_86/batchnorm/RsqrtRsqrt(batch_normalization_86/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_86/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_86_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_86/batchnorm/mulMul*batch_normalization_86/batchnorm/Rsqrt:y:0;batch_normalization_86/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_86/batchnorm/mul_1Muldense_124/BiasAdd:output:0(batch_normalization_86/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
&batch_normalization_86/batchnorm/mul_2Mul/batch_normalization_86/moments/Squeeze:output:0(batch_normalization_86/batchnorm/mul:z:0*
T0*
_output_shapes
:�
/batch_normalization_86/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_86_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_86/batchnorm/subSub7batch_normalization_86/batchnorm/ReadVariableOp:value:0*batch_normalization_86/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_86/batchnorm/add_1AddV2*batch_normalization_86/batchnorm/mul_1:z:0(batch_normalization_86/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������s
re_lu_44/ReluRelu*batch_normalization_86/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������]
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_10/dropout/MulMulre_lu_44/Relu:activations:0!dropout_10/dropout/Const:output:0*
T0*'
_output_shapes
:���������c
dropout_10/dropout/ShapeShapere_lu_44/Relu:activations:0*
T0*
_output_shapes
:�
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0f
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*'
_output_shapes
:����������
dense_125/MatMul/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_125/MatMulMatMuldropout_10/dropout/Mul_1:z:0'dense_125/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_125/BiasAdd/ReadVariableOpReadVariableOp)dense_125_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_125/BiasAddBiasAdddense_125/MatMul:product:0(dense_125/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@
5batch_normalization_87/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_87/moments/meanMeandense_125/BiasAdd:output:0>batch_normalization_87/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
+batch_normalization_87/moments/StopGradientStopGradient,batch_normalization_87/moments/mean:output:0*
T0*
_output_shapes

:@�
0batch_normalization_87/moments/SquaredDifferenceSquaredDifferencedense_125/BiasAdd:output:04batch_normalization_87/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������@�
9batch_normalization_87/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_87/moments/varianceMean4batch_normalization_87/moments/SquaredDifference:z:0Bbatch_normalization_87/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
&batch_normalization_87/moments/SqueezeSqueeze,batch_normalization_87/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 �
(batch_normalization_87/moments/Squeeze_1Squeeze0batch_normalization_87/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 q
,batch_normalization_87/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_87/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_87_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
*batch_normalization_87/AssignMovingAvg/subSub=batch_normalization_87/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_87/moments/Squeeze:output:0*
T0*
_output_shapes
:@�
*batch_normalization_87/AssignMovingAvg/mulMul.batch_normalization_87/AssignMovingAvg/sub:z:05batch_normalization_87/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
&batch_normalization_87/AssignMovingAvgAssignSubVariableOp>batch_normalization_87_assignmovingavg_readvariableop_resource.batch_normalization_87/AssignMovingAvg/mul:z:06^batch_normalization_87/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_87/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_87/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_87_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
,batch_normalization_87/AssignMovingAvg_1/subSub?batch_normalization_87/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_87/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@�
,batch_normalization_87/AssignMovingAvg_1/mulMul0batch_normalization_87/AssignMovingAvg_1/sub:z:07batch_normalization_87/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
(batch_normalization_87/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_87_assignmovingavg_1_readvariableop_resource0batch_normalization_87/AssignMovingAvg_1/mul:z:08^batch_normalization_87/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_87/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_87/batchnorm/addAddV21batch_normalization_87/moments/Squeeze_1:output:0/batch_normalization_87/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_87/batchnorm/RsqrtRsqrt(batch_normalization_87/batchnorm/add:z:0*
T0*
_output_shapes
:@�
3batch_normalization_87/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_87_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_87/batchnorm/mulMul*batch_normalization_87/batchnorm/Rsqrt:y:0;batch_normalization_87/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_87/batchnorm/mul_1Muldense_125/BiasAdd:output:0(batch_normalization_87/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
&batch_normalization_87/batchnorm/mul_2Mul/batch_normalization_87/moments/Squeeze:output:0(batch_normalization_87/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
/batch_normalization_87/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_87_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_87/batchnorm/subSub7batch_normalization_87/batchnorm/ReadVariableOp:value:0*batch_normalization_87/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_87/batchnorm/add_1AddV2*batch_normalization_87/batchnorm/mul_1:z:0(batch_normalization_87/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@s
re_lu_45/ReluRelu*batch_normalization_87/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@_
dropout_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_10/dropout_1/MulMulre_lu_45/Relu:activations:0#dropout_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:���������@e
dropout_10/dropout_1/ShapeShapere_lu_45/Relu:activations:0*
T0*
_output_shapes
:�
1dropout_10/dropout_1/random_uniform/RandomUniformRandomUniform#dropout_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0h
#dropout_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
!dropout_10/dropout_1/GreaterEqualGreaterEqual:dropout_10/dropout_1/random_uniform/RandomUniform:output:0,dropout_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
dropout_10/dropout_1/CastCast%dropout_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
dropout_10/dropout_1/Mul_1Muldropout_10/dropout_1/Mul:z:0dropout_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:���������@�
dense_126/MatMul/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_126/MatMulMatMuldropout_10/dropout_1/Mul_1:z:0'dense_126/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_126/BiasAdd/ReadVariableOpReadVariableOp)dense_126_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_126/BiasAddBiasAdddense_126/MatMul:product:0(dense_126/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@
5batch_normalization_88/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_88/moments/meanMeandense_126/BiasAdd:output:0>batch_normalization_88/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
+batch_normalization_88/moments/StopGradientStopGradient,batch_normalization_88/moments/mean:output:0*
T0*
_output_shapes

:@�
0batch_normalization_88/moments/SquaredDifferenceSquaredDifferencedense_126/BiasAdd:output:04batch_normalization_88/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������@�
9batch_normalization_88/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_88/moments/varianceMean4batch_normalization_88/moments/SquaredDifference:z:0Bbatch_normalization_88/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
&batch_normalization_88/moments/SqueezeSqueeze,batch_normalization_88/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 �
(batch_normalization_88/moments/Squeeze_1Squeeze0batch_normalization_88/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 q
,batch_normalization_88/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_88/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_88_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
*batch_normalization_88/AssignMovingAvg/subSub=batch_normalization_88/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_88/moments/Squeeze:output:0*
T0*
_output_shapes
:@�
*batch_normalization_88/AssignMovingAvg/mulMul.batch_normalization_88/AssignMovingAvg/sub:z:05batch_normalization_88/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
&batch_normalization_88/AssignMovingAvgAssignSubVariableOp>batch_normalization_88_assignmovingavg_readvariableop_resource.batch_normalization_88/AssignMovingAvg/mul:z:06^batch_normalization_88/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_88/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_88/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_88_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
,batch_normalization_88/AssignMovingAvg_1/subSub?batch_normalization_88/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_88/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@�
,batch_normalization_88/AssignMovingAvg_1/mulMul0batch_normalization_88/AssignMovingAvg_1/sub:z:07batch_normalization_88/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
(batch_normalization_88/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_88_assignmovingavg_1_readvariableop_resource0batch_normalization_88/AssignMovingAvg_1/mul:z:08^batch_normalization_88/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_88/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_88/batchnorm/addAddV21batch_normalization_88/moments/Squeeze_1:output:0/batch_normalization_88/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_88/batchnorm/RsqrtRsqrt(batch_normalization_88/batchnorm/add:z:0*
T0*
_output_shapes
:@�
3batch_normalization_88/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_88_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_88/batchnorm/mulMul*batch_normalization_88/batchnorm/Rsqrt:y:0;batch_normalization_88/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_88/batchnorm/mul_1Muldense_126/BiasAdd:output:0(batch_normalization_88/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
&batch_normalization_88/batchnorm/mul_2Mul/batch_normalization_88/moments/Squeeze:output:0(batch_normalization_88/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
/batch_normalization_88/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_88_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_88/batchnorm/subSub7batch_normalization_88/batchnorm/ReadVariableOp:value:0*batch_normalization_88/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_88/batchnorm/add_1AddV2*batch_normalization_88/batchnorm/mul_1:z:0(batch_normalization_88/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@s
re_lu_46/ReluRelu*batch_normalization_88/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@�
dense_127/MatMul/ReadVariableOpReadVariableOp(dense_127_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_127/MatMulMatMulre_lu_46/Relu:activations:0'dense_127/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_127/BiasAdd/ReadVariableOpReadVariableOp)dense_127_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_127/BiasAddBiasAdddense_127/MatMul:product:0(dense_127/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
#dense_124/kernel/Regularizer/L2LossL2Loss:dense_124/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_124/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_124/kernel/Regularizer/mulMul+dense_124/kernel/Regularizer/mul/x:output:0,dense_124/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
#dense_125/kernel/Regularizer/L2LossL2Loss:dense_125/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_125/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_125/kernel/Regularizer/mulMul+dense_125/kernel/Regularizer/mul/x:output:0,dense_125/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
#dense_126/kernel/Regularizer/L2LossL2Loss:dense_126/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_126/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_126/kernel/Regularizer/mulMul+dense_126/kernel/Regularizer/mul/x:output:0,dense_126/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_127/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^batch_normalization_86/AssignMovingAvg6^batch_normalization_86/AssignMovingAvg/ReadVariableOp)^batch_normalization_86/AssignMovingAvg_18^batch_normalization_86/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_86/batchnorm/ReadVariableOp4^batch_normalization_86/batchnorm/mul/ReadVariableOp'^batch_normalization_87/AssignMovingAvg6^batch_normalization_87/AssignMovingAvg/ReadVariableOp)^batch_normalization_87/AssignMovingAvg_18^batch_normalization_87/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_87/batchnorm/ReadVariableOp4^batch_normalization_87/batchnorm/mul/ReadVariableOp'^batch_normalization_88/AssignMovingAvg6^batch_normalization_88/AssignMovingAvg/ReadVariableOp)^batch_normalization_88/AssignMovingAvg_18^batch_normalization_88/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_88/batchnorm/ReadVariableOp4^batch_normalization_88/batchnorm/mul/ReadVariableOp!^dense_124/BiasAdd/ReadVariableOp ^dense_124/MatMul/ReadVariableOp3^dense_124/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_125/BiasAdd/ReadVariableOp ^dense_125/MatMul/ReadVariableOp3^dense_125/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_126/BiasAdd/ReadVariableOp ^dense_126/MatMul/ReadVariableOp3^dense_126/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_127/BiasAdd/ReadVariableOp ^dense_127/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������<Z: : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_86/AssignMovingAvg&batch_normalization_86/AssignMovingAvg2n
5batch_normalization_86/AssignMovingAvg/ReadVariableOp5batch_normalization_86/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_86/AssignMovingAvg_1(batch_normalization_86/AssignMovingAvg_12r
7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_86/batchnorm/ReadVariableOp/batch_normalization_86/batchnorm/ReadVariableOp2j
3batch_normalization_86/batchnorm/mul/ReadVariableOp3batch_normalization_86/batchnorm/mul/ReadVariableOp2P
&batch_normalization_87/AssignMovingAvg&batch_normalization_87/AssignMovingAvg2n
5batch_normalization_87/AssignMovingAvg/ReadVariableOp5batch_normalization_87/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_87/AssignMovingAvg_1(batch_normalization_87/AssignMovingAvg_12r
7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_87/batchnorm/ReadVariableOp/batch_normalization_87/batchnorm/ReadVariableOp2j
3batch_normalization_87/batchnorm/mul/ReadVariableOp3batch_normalization_87/batchnorm/mul/ReadVariableOp2P
&batch_normalization_88/AssignMovingAvg&batch_normalization_88/AssignMovingAvg2n
5batch_normalization_88/AssignMovingAvg/ReadVariableOp5batch_normalization_88/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_88/AssignMovingAvg_1(batch_normalization_88/AssignMovingAvg_12r
7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_88/batchnorm/ReadVariableOp/batch_normalization_88/batchnorm/ReadVariableOp2j
3batch_normalization_88/batchnorm/mul/ReadVariableOp3batch_normalization_88/batchnorm/mul/ReadVariableOp2D
 dense_124/BiasAdd/ReadVariableOp dense_124/BiasAdd/ReadVariableOp2B
dense_124/MatMul/ReadVariableOpdense_124/MatMul/ReadVariableOp2h
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_125/BiasAdd/ReadVariableOp dense_125/BiasAdd/ReadVariableOp2B
dense_125/MatMul/ReadVariableOpdense_125/MatMul/ReadVariableOp2h
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_126/BiasAdd/ReadVariableOp dense_126/BiasAdd/ReadVariableOp2B
dense_126/MatMul/ReadVariableOpdense_126/MatMul/ReadVariableOp2h
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_127/BiasAdd/ReadVariableOp dense_127/BiasAdd/ReadVariableOp2B
dense_127/MatMul/ReadVariableOpdense_127/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������<Z
 
_user_specified_nameinputs
�
b
F__inference_re_lu_46_layer_call_and_return_conditional_losses_14105225

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:���������@Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_88_layer_call_fn_14105127

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
G__inference_dense_125_layer_call_and_return_conditional_losses_14104961

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2dense_125/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
#dense_125/kernel/Regularizer/L2LossL2Loss:dense_125/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_125/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_125/kernel/Regularizer/mulMul+dense_125/kernel/Regularizer/mul/x:output:0,dense_125/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_125/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
+__inference_model_16_layer_call_fn_14104461

inputs;
(dense_124_matmul_readvariableop_resource:	�*7
)dense_124_biasadd_readvariableop_resource:L
>batch_normalization_86_assignmovingavg_readvariableop_resource:N
@batch_normalization_86_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_86_batchnorm_mul_readvariableop_resource:F
8batch_normalization_86_batchnorm_readvariableop_resource::
(dense_125_matmul_readvariableop_resource:@7
)dense_125_biasadd_readvariableop_resource:@L
>batch_normalization_87_assignmovingavg_readvariableop_resource:@N
@batch_normalization_87_assignmovingavg_1_readvariableop_resource:@J
<batch_normalization_87_batchnorm_mul_readvariableop_resource:@F
8batch_normalization_87_batchnorm_readvariableop_resource:@:
(dense_126_matmul_readvariableop_resource:@@7
)dense_126_biasadd_readvariableop_resource:@L
>batch_normalization_88_assignmovingavg_readvariableop_resource:@N
@batch_normalization_88_assignmovingavg_1_readvariableop_resource:@J
<batch_normalization_88_batchnorm_mul_readvariableop_resource:@F
8batch_normalization_88_batchnorm_readvariableop_resource:@:
(dense_127_matmul_readvariableop_resource:@7
)dense_127_biasadd_readvariableop_resource:
identity��&batch_normalization_86/AssignMovingAvg�5batch_normalization_86/AssignMovingAvg/ReadVariableOp�(batch_normalization_86/AssignMovingAvg_1�7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_86/batchnorm/ReadVariableOp�3batch_normalization_86/batchnorm/mul/ReadVariableOp�&batch_normalization_87/AssignMovingAvg�5batch_normalization_87/AssignMovingAvg/ReadVariableOp�(batch_normalization_87/AssignMovingAvg_1�7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_87/batchnorm/ReadVariableOp�3batch_normalization_87/batchnorm/mul/ReadVariableOp�&batch_normalization_88/AssignMovingAvg�5batch_normalization_88/AssignMovingAvg/ReadVariableOp�(batch_normalization_88/AssignMovingAvg_1�7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_88/batchnorm/ReadVariableOp�3batch_normalization_88/batchnorm/mul/ReadVariableOp� dense_124/BiasAdd/ReadVariableOp�dense_124/MatMul/ReadVariableOp�2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp� dense_125/BiasAdd/ReadVariableOp�dense_125/MatMul/ReadVariableOp�2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp� dense_126/BiasAdd/ReadVariableOp�dense_126/MatMul/ReadVariableOp�2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp� dense_127/BiasAdd/ReadVariableOp�dense_127/MatMul/ReadVariableOpa
flatten_39/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  s
flatten_39/ReshapeReshapeinputsflatten_39/Const:output:0*
T0*(
_output_shapes
:����������*�
dense_124/MatMul/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
dense_124/MatMulMatMulflatten_39/Reshape:output:0'dense_124/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_124/BiasAdd/ReadVariableOpReadVariableOp)dense_124_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_124/BiasAddBiasAdddense_124/MatMul:product:0(dense_124/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
5batch_normalization_86/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_86/moments/meanMeandense_124/BiasAdd:output:0>batch_normalization_86/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
+batch_normalization_86/moments/StopGradientStopGradient,batch_normalization_86/moments/mean:output:0*
T0*
_output_shapes

:�
0batch_normalization_86/moments/SquaredDifferenceSquaredDifferencedense_124/BiasAdd:output:04batch_normalization_86/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
9batch_normalization_86/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_86/moments/varianceMean4batch_normalization_86/moments/SquaredDifference:z:0Bbatch_normalization_86/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
&batch_normalization_86/moments/SqueezeSqueeze,batch_normalization_86/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
(batch_normalization_86/moments/Squeeze_1Squeeze0batch_normalization_86/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_86/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_86/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_86_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_86/AssignMovingAvg/subSub=batch_normalization_86/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_86/moments/Squeeze:output:0*
T0*
_output_shapes
:�
*batch_normalization_86/AssignMovingAvg/mulMul.batch_normalization_86/AssignMovingAvg/sub:z:05batch_normalization_86/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
&batch_normalization_86/AssignMovingAvgAssignSubVariableOp>batch_normalization_86_assignmovingavg_readvariableop_resource.batch_normalization_86/AssignMovingAvg/mul:z:06^batch_normalization_86/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_86/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_86/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_86_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_86/AssignMovingAvg_1/subSub?batch_normalization_86/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_86/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
,batch_normalization_86/AssignMovingAvg_1/mulMul0batch_normalization_86/AssignMovingAvg_1/sub:z:07batch_normalization_86/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_86/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_86_assignmovingavg_1_readvariableop_resource0batch_normalization_86/AssignMovingAvg_1/mul:z:08^batch_normalization_86/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_86/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_86/batchnorm/addAddV21batch_normalization_86/moments/Squeeze_1:output:0/batch_normalization_86/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_86/batchnorm/RsqrtRsqrt(batch_normalization_86/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_86/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_86_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_86/batchnorm/mulMul*batch_normalization_86/batchnorm/Rsqrt:y:0;batch_normalization_86/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_86/batchnorm/mul_1Muldense_124/BiasAdd:output:0(batch_normalization_86/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
&batch_normalization_86/batchnorm/mul_2Mul/batch_normalization_86/moments/Squeeze:output:0(batch_normalization_86/batchnorm/mul:z:0*
T0*
_output_shapes
:�
/batch_normalization_86/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_86_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_86/batchnorm/subSub7batch_normalization_86/batchnorm/ReadVariableOp:value:0*batch_normalization_86/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_86/batchnorm/add_1AddV2*batch_normalization_86/batchnorm/mul_1:z:0(batch_normalization_86/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������s
re_lu_44/ReluRelu*batch_normalization_86/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������]
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_10/dropout/MulMulre_lu_44/Relu:activations:0!dropout_10/dropout/Const:output:0*
T0*'
_output_shapes
:���������c
dropout_10/dropout/ShapeShapere_lu_44/Relu:activations:0*
T0*
_output_shapes
:�
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0f
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*'
_output_shapes
:����������
dense_125/MatMul/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_125/MatMulMatMuldropout_10/dropout/Mul_1:z:0'dense_125/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_125/BiasAdd/ReadVariableOpReadVariableOp)dense_125_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_125/BiasAddBiasAdddense_125/MatMul:product:0(dense_125/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@
5batch_normalization_87/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_87/moments/meanMeandense_125/BiasAdd:output:0>batch_normalization_87/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
+batch_normalization_87/moments/StopGradientStopGradient,batch_normalization_87/moments/mean:output:0*
T0*
_output_shapes

:@�
0batch_normalization_87/moments/SquaredDifferenceSquaredDifferencedense_125/BiasAdd:output:04batch_normalization_87/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������@�
9batch_normalization_87/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_87/moments/varianceMean4batch_normalization_87/moments/SquaredDifference:z:0Bbatch_normalization_87/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
&batch_normalization_87/moments/SqueezeSqueeze,batch_normalization_87/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 �
(batch_normalization_87/moments/Squeeze_1Squeeze0batch_normalization_87/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 q
,batch_normalization_87/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_87/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_87_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
*batch_normalization_87/AssignMovingAvg/subSub=batch_normalization_87/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_87/moments/Squeeze:output:0*
T0*
_output_shapes
:@�
*batch_normalization_87/AssignMovingAvg/mulMul.batch_normalization_87/AssignMovingAvg/sub:z:05batch_normalization_87/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
&batch_normalization_87/AssignMovingAvgAssignSubVariableOp>batch_normalization_87_assignmovingavg_readvariableop_resource.batch_normalization_87/AssignMovingAvg/mul:z:06^batch_normalization_87/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_87/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_87/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_87_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
,batch_normalization_87/AssignMovingAvg_1/subSub?batch_normalization_87/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_87/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@�
,batch_normalization_87/AssignMovingAvg_1/mulMul0batch_normalization_87/AssignMovingAvg_1/sub:z:07batch_normalization_87/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
(batch_normalization_87/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_87_assignmovingavg_1_readvariableop_resource0batch_normalization_87/AssignMovingAvg_1/mul:z:08^batch_normalization_87/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_87/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_87/batchnorm/addAddV21batch_normalization_87/moments/Squeeze_1:output:0/batch_normalization_87/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_87/batchnorm/RsqrtRsqrt(batch_normalization_87/batchnorm/add:z:0*
T0*
_output_shapes
:@�
3batch_normalization_87/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_87_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_87/batchnorm/mulMul*batch_normalization_87/batchnorm/Rsqrt:y:0;batch_normalization_87/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_87/batchnorm/mul_1Muldense_125/BiasAdd:output:0(batch_normalization_87/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
&batch_normalization_87/batchnorm/mul_2Mul/batch_normalization_87/moments/Squeeze:output:0(batch_normalization_87/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
/batch_normalization_87/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_87_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_87/batchnorm/subSub7batch_normalization_87/batchnorm/ReadVariableOp:value:0*batch_normalization_87/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_87/batchnorm/add_1AddV2*batch_normalization_87/batchnorm/mul_1:z:0(batch_normalization_87/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@s
re_lu_45/ReluRelu*batch_normalization_87/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@_
dropout_10/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_10/dropout_1/MulMulre_lu_45/Relu:activations:0#dropout_10/dropout_1/Const:output:0*
T0*'
_output_shapes
:���������@e
dropout_10/dropout_1/ShapeShapere_lu_45/Relu:activations:0*
T0*
_output_shapes
:�
1dropout_10/dropout_1/random_uniform/RandomUniformRandomUniform#dropout_10/dropout_1/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0h
#dropout_10/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
!dropout_10/dropout_1/GreaterEqualGreaterEqual:dropout_10/dropout_1/random_uniform/RandomUniform:output:0,dropout_10/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
dropout_10/dropout_1/CastCast%dropout_10/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
dropout_10/dropout_1/Mul_1Muldropout_10/dropout_1/Mul:z:0dropout_10/dropout_1/Cast:y:0*
T0*'
_output_shapes
:���������@�
dense_126/MatMul/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_126/MatMulMatMuldropout_10/dropout_1/Mul_1:z:0'dense_126/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 dense_126/BiasAdd/ReadVariableOpReadVariableOp)dense_126_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_126/BiasAddBiasAdddense_126/MatMul:product:0(dense_126/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@
5batch_normalization_88/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_88/moments/meanMeandense_126/BiasAdd:output:0>batch_normalization_88/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
+batch_normalization_88/moments/StopGradientStopGradient,batch_normalization_88/moments/mean:output:0*
T0*
_output_shapes

:@�
0batch_normalization_88/moments/SquaredDifferenceSquaredDifferencedense_126/BiasAdd:output:04batch_normalization_88/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������@�
9batch_normalization_88/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_88/moments/varianceMean4batch_normalization_88/moments/SquaredDifference:z:0Bbatch_normalization_88/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
&batch_normalization_88/moments/SqueezeSqueeze,batch_normalization_88/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 �
(batch_normalization_88/moments/Squeeze_1Squeeze0batch_normalization_88/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 q
,batch_normalization_88/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_88/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_88_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
*batch_normalization_88/AssignMovingAvg/subSub=batch_normalization_88/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_88/moments/Squeeze:output:0*
T0*
_output_shapes
:@�
*batch_normalization_88/AssignMovingAvg/mulMul.batch_normalization_88/AssignMovingAvg/sub:z:05batch_normalization_88/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
&batch_normalization_88/AssignMovingAvgAssignSubVariableOp>batch_normalization_88_assignmovingavg_readvariableop_resource.batch_normalization_88/AssignMovingAvg/mul:z:06^batch_normalization_88/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_88/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_88/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_88_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
,batch_normalization_88/AssignMovingAvg_1/subSub?batch_normalization_88/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_88/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@�
,batch_normalization_88/AssignMovingAvg_1/mulMul0batch_normalization_88/AssignMovingAvg_1/sub:z:07batch_normalization_88/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
(batch_normalization_88/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_88_assignmovingavg_1_readvariableop_resource0batch_normalization_88/AssignMovingAvg_1/mul:z:08^batch_normalization_88/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_88/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_88/batchnorm/addAddV21batch_normalization_88/moments/Squeeze_1:output:0/batch_normalization_88/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_88/batchnorm/RsqrtRsqrt(batch_normalization_88/batchnorm/add:z:0*
T0*
_output_shapes
:@�
3batch_normalization_88/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_88_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_88/batchnorm/mulMul*batch_normalization_88/batchnorm/Rsqrt:y:0;batch_normalization_88/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
&batch_normalization_88/batchnorm/mul_1Muldense_126/BiasAdd:output:0(batch_normalization_88/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
&batch_normalization_88/batchnorm/mul_2Mul/batch_normalization_88/moments/Squeeze:output:0(batch_normalization_88/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
/batch_normalization_88/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_88_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization_88/batchnorm/subSub7batch_normalization_88/batchnorm/ReadVariableOp:value:0*batch_normalization_88/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
&batch_normalization_88/batchnorm/add_1AddV2*batch_normalization_88/batchnorm/mul_1:z:0(batch_normalization_88/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@s
re_lu_46/ReluRelu*batch_normalization_88/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������@�
dense_127/MatMul/ReadVariableOpReadVariableOp(dense_127_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_127/MatMulMatMulre_lu_46/Relu:activations:0'dense_127/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_127/BiasAdd/ReadVariableOpReadVariableOp)dense_127_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_127/BiasAddBiasAdddense_127/MatMul:product:0(dense_127/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_124_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
#dense_124/kernel/Regularizer/L2LossL2Loss:dense_124/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_124/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_124/kernel/Regularizer/mulMul+dense_124/kernel/Regularizer/mul/x:output:0,dense_124/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_125_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
#dense_125/kernel/Regularizer/L2LossL2Loss:dense_125/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_125/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_125/kernel/Regularizer/mulMul+dense_125/kernel/Regularizer/mul/x:output:0,dense_125/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_126_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
#dense_126/kernel/Regularizer/L2LossL2Loss:dense_126/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_126/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_126/kernel/Regularizer/mulMul+dense_126/kernel/Regularizer/mul/x:output:0,dense_126/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_127/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^batch_normalization_86/AssignMovingAvg6^batch_normalization_86/AssignMovingAvg/ReadVariableOp)^batch_normalization_86/AssignMovingAvg_18^batch_normalization_86/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_86/batchnorm/ReadVariableOp4^batch_normalization_86/batchnorm/mul/ReadVariableOp'^batch_normalization_87/AssignMovingAvg6^batch_normalization_87/AssignMovingAvg/ReadVariableOp)^batch_normalization_87/AssignMovingAvg_18^batch_normalization_87/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_87/batchnorm/ReadVariableOp4^batch_normalization_87/batchnorm/mul/ReadVariableOp'^batch_normalization_88/AssignMovingAvg6^batch_normalization_88/AssignMovingAvg/ReadVariableOp)^batch_normalization_88/AssignMovingAvg_18^batch_normalization_88/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_88/batchnorm/ReadVariableOp4^batch_normalization_88/batchnorm/mul/ReadVariableOp!^dense_124/BiasAdd/ReadVariableOp ^dense_124/MatMul/ReadVariableOp3^dense_124/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_125/BiasAdd/ReadVariableOp ^dense_125/MatMul/ReadVariableOp3^dense_125/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_126/BiasAdd/ReadVariableOp ^dense_126/MatMul/ReadVariableOp3^dense_126/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_127/BiasAdd/ReadVariableOp ^dense_127/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������<Z: : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_86/AssignMovingAvg&batch_normalization_86/AssignMovingAvg2n
5batch_normalization_86/AssignMovingAvg/ReadVariableOp5batch_normalization_86/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_86/AssignMovingAvg_1(batch_normalization_86/AssignMovingAvg_12r
7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp7batch_normalization_86/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_86/batchnorm/ReadVariableOp/batch_normalization_86/batchnorm/ReadVariableOp2j
3batch_normalization_86/batchnorm/mul/ReadVariableOp3batch_normalization_86/batchnorm/mul/ReadVariableOp2P
&batch_normalization_87/AssignMovingAvg&batch_normalization_87/AssignMovingAvg2n
5batch_normalization_87/AssignMovingAvg/ReadVariableOp5batch_normalization_87/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_87/AssignMovingAvg_1(batch_normalization_87/AssignMovingAvg_12r
7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp7batch_normalization_87/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_87/batchnorm/ReadVariableOp/batch_normalization_87/batchnorm/ReadVariableOp2j
3batch_normalization_87/batchnorm/mul/ReadVariableOp3batch_normalization_87/batchnorm/mul/ReadVariableOp2P
&batch_normalization_88/AssignMovingAvg&batch_normalization_88/AssignMovingAvg2n
5batch_normalization_88/AssignMovingAvg/ReadVariableOp5batch_normalization_88/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_88/AssignMovingAvg_1(batch_normalization_88/AssignMovingAvg_12r
7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp7batch_normalization_88/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_88/batchnorm/ReadVariableOp/batch_normalization_88/batchnorm/ReadVariableOp2j
3batch_normalization_88/batchnorm/mul/ReadVariableOp3batch_normalization_88/batchnorm/mul/ReadVariableOp2D
 dense_124/BiasAdd/ReadVariableOp dense_124/BiasAdd/ReadVariableOp2B
dense_124/MatMul/ReadVariableOpdense_124/MatMul/ReadVariableOp2h
2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2dense_124/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_125/BiasAdd/ReadVariableOp dense_125/BiasAdd/ReadVariableOp2B
dense_125/MatMul/ReadVariableOpdense_125/MatMul/ReadVariableOp2h
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_126/BiasAdd/ReadVariableOp dense_126/BiasAdd/ReadVariableOp2B
dense_126/MatMul/ReadVariableOpdense_126/MatMul/ReadVariableOp2h
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_127/BiasAdd/ReadVariableOp dense_127/BiasAdd/ReadVariableOp2B
dense_127/MatMul/ReadVariableOpdense_127/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������<Z
 
_user_specified_nameinputs
�
�
&__inference_signature_wrapper_14104203	
image
unknown:	�*
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallimageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *,
f'R%
#__inference__wrapped_model_14102496o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������<Z: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:���������<Z

_user_specified_nameimage
�
b
F__inference_re_lu_45_layer_call_and_return_conditional_losses_14105079

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:���������@Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
,__inference_dense_125_layer_call_fn_14104947

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2dense_125/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
#dense_125/kernel/Regularizer/L2LossL2Loss:dense_125/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_125/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_125/kernel/Regularizer/mulMul+dense_125/kernel/Regularizer/mul/x:output:0,dense_125/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_125/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp2dense_125/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_re_lu_44_layer_call_and_return_conditional_losses_14104865

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:���������Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
L
-__inference_dropout_10_layer_call_fn_14104882

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
g
H__inference_dropout_10_layer_call_and_return_conditional_losses_14104916

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_2_14105272M
;dense_126_kernel_regularizer_l2loss_readvariableop_resource:@@
identity��2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp�
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp;dense_126_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:@@*
dtype0�
#dense_126/kernel/Regularizer/L2LossL2Loss:dense_126/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_126/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_126/kernel/Regularizer/mulMul+dense_126/kernel/Regularizer/mul/x:output:0,dense_126/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_126/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^dense_126/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp
�
f
H__inference_dropout_10_layer_call_and_return_conditional_losses_14104921

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
H__inference_dropout_10_layer_call_and_return_conditional_losses_14104904

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
G__inference_dense_126_layer_call_and_return_conditional_losses_14105107

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2dense_126/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
#dense_126/kernel/Regularizer/L2LossL2Loss:dense_126/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_126/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
 dense_126/kernel/Regularizer/mulMul+dense_126/kernel/Regularizer/mul/x:output:0,dense_126/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_126/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp2dense_126/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
g
H__inference_dropout_10_layer_call_and_return_conditional_losses_14104933

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_87_layer_call_and_return_conditional_losses_14105035

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs"�	-
saver_filename:0
Identity:0Identity_558"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
image6
serving_default_image:0���������<Z=
	dense_1270
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+axis
	,gamma
-beta
.moving_mean
/moving_variance"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses
<_random_generator"
_tf_keras_layer
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

Ckernel
Dbias"
_tf_keras_layer
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance"
_tf_keras_layer
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias"
_tf_keras_layer
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
daxis
	egamma
fbeta
gmoving_mean
hmoving_variance"
_tf_keras_layer
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

ukernel
vbias"
_tf_keras_layer
�
#0
$1
,2
-3
.4
/5
C6
D7
L8
M9
N10
O11
\12
]13
e14
f15
g16
h17
u18
v19"
trackable_list_wrapper
�
#0
$1
,2
-3
C4
D5
L6
M7
\8
]9
e10
f11
u12
v13"
trackable_list_wrapper
5
w0
x1
y2"
trackable_list_wrapper
�
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
�trace_1
�trace_2
�trace_32�
+__inference_model_16_layer_call_fn_14102934
+__inference_model_16_layer_call_fn_14104310
+__inference_model_16_layer_call_fn_14104461
+__inference_model_16_layer_call_fn_14103870�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
F__inference_model_16_layer_call_and_return_conditional_losses_14104556
F__inference_model_16_layer_call_and_return_conditional_losses_14104707
F__inference_model_16_layer_call_and_return_conditional_losses_14103977
F__inference_model_16_layer_call_and_return_conditional_losses_14104140�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
#__inference__wrapped_model_14102496image"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
	�iter
�beta_1
�beta_2

�decay#m�$m�,m�-m�Cm�Dm�Lm�Mm�\m�]m�em�fm�um�vm�#v�$v�,v�-v�Cv�Dv�Lv�Mv�\v�]v�ev�fv�uv�vv�"
	optimizer
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_flatten_39_layer_call_fn_14104713�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_flatten_39_layer_call_and_return_conditional_losses_14104719�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
'
w0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_dense_124_layer_call_fn_14104733�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_dense_124_layer_call_and_return_conditional_losses_14104747�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!	�*2dense_124/kernel
:2dense_124/bias
<
,0
-1
.2
/3"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_86_layer_call_fn_14104767
9__inference_batch_normalization_86_layer_call_fn_14104801�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_batch_normalization_86_layer_call_and_return_conditional_losses_14104821
T__inference_batch_normalization_86_layer_call_and_return_conditional_losses_14104855�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(2batch_normalization_86/gamma
):'2batch_normalization_86/beta
2:0 (2"batch_normalization_86/moving_mean
6:4 (2&batch_normalization_86/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_re_lu_44_layer_call_fn_14104860�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_re_lu_44_layer_call_and_return_conditional_losses_14104865�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
-__inference_dropout_10_layer_call_fn_14104870
-__inference_dropout_10_layer_call_fn_14104882
-__inference_dropout_10_layer_call_fn_14104887
-__inference_dropout_10_layer_call_fn_14104899�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
H__inference_dropout_10_layer_call_and_return_conditional_losses_14104904
H__inference_dropout_10_layer_call_and_return_conditional_losses_14104916
H__inference_dropout_10_layer_call_and_return_conditional_losses_14104921
H__inference_dropout_10_layer_call_and_return_conditional_losses_14104933�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
"
_generic_user_object
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
'
x0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_dense_125_layer_call_fn_14104947�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_dense_125_layer_call_and_return_conditional_losses_14104961�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": @2dense_125/kernel
:@2dense_125/bias
<
L0
M1
N2
O3"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_87_layer_call_fn_14104981
9__inference_batch_normalization_87_layer_call_fn_14105015�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_batch_normalization_87_layer_call_and_return_conditional_losses_14105035
T__inference_batch_normalization_87_layer_call_and_return_conditional_losses_14105069�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_87/gamma
):'@2batch_normalization_87/beta
2:0@ (2"batch_normalization_87/moving_mean
6:4@ (2&batch_normalization_87/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_re_lu_45_layer_call_fn_14105074�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_re_lu_45_layer_call_and_return_conditional_losses_14105079�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
'
y0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_dense_126_layer_call_fn_14105093�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_dense_126_layer_call_and_return_conditional_losses_14105107�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": @@2dense_126/kernel
:@2dense_126/bias
<
e0
f1
g2
h3"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_88_layer_call_fn_14105127
9__inference_batch_normalization_88_layer_call_fn_14105161�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_batch_normalization_88_layer_call_and_return_conditional_losses_14105181
T__inference_batch_normalization_88_layer_call_and_return_conditional_losses_14105215�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_88/gamma
):'@2batch_normalization_88/beta
2:0@ (2"batch_normalization_88/moving_mean
6:4@ (2&batch_normalization_88/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_re_lu_46_layer_call_fn_14105220�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_re_lu_46_layer_call_and_return_conditional_losses_14105225�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_dense_127_layer_call_fn_14105235�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_dense_127_layer_call_and_return_conditional_losses_14105245�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": @2dense_127/kernel
:2dense_127/bias
�
�trace_02�
__inference_loss_fn_0_14105254�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_14105263�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_14105272�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
J
.0
/1
N2
O3
g4
h5"
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_model_16_layer_call_fn_14102934image"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_model_16_layer_call_fn_14104310inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_model_16_layer_call_fn_14104461inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_model_16_layer_call_fn_14103870image"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_model_16_layer_call_and_return_conditional_losses_14104556inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_model_16_layer_call_and_return_conditional_losses_14104707inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_model_16_layer_call_and_return_conditional_losses_14103977image"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_model_16_layer_call_and_return_conditional_losses_14104140image"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
�B�
&__inference_signature_wrapper_14104203image"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_flatten_39_layer_call_fn_14104713inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_flatten_39_layer_call_and_return_conditional_losses_14104719inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
w0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dense_124_layer_call_fn_14104733inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dense_124_layer_call_and_return_conditional_losses_14104747inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_batch_normalization_86_layer_call_fn_14104767inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
9__inference_batch_normalization_86_layer_call_fn_14104801inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_86_layer_call_and_return_conditional_losses_14104821inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_86_layer_call_and_return_conditional_losses_14104855inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_re_lu_44_layer_call_fn_14104860inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_re_lu_44_layer_call_and_return_conditional_losses_14104865inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dropout_10_layer_call_fn_14104870inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_dropout_10_layer_call_fn_14104882inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_dropout_10_layer_call_fn_14104887inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_dropout_10_layer_call_fn_14104899inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_10_layer_call_and_return_conditional_losses_14104904inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_10_layer_call_and_return_conditional_losses_14104916inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_10_layer_call_and_return_conditional_losses_14104921inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_10_layer_call_and_return_conditional_losses_14104933inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
x0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dense_125_layer_call_fn_14104947inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dense_125_layer_call_and_return_conditional_losses_14104961inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_batch_normalization_87_layer_call_fn_14104981inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
9__inference_batch_normalization_87_layer_call_fn_14105015inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_87_layer_call_and_return_conditional_losses_14105035inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_87_layer_call_and_return_conditional_losses_14105069inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_re_lu_45_layer_call_fn_14105074inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_re_lu_45_layer_call_and_return_conditional_losses_14105079inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
y0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dense_126_layer_call_fn_14105093inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dense_126_layer_call_and_return_conditional_losses_14105107inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_batch_normalization_88_layer_call_fn_14105127inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
9__inference_batch_normalization_88_layer_call_fn_14105161inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_88_layer_call_and_return_conditional_losses_14105181inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_88_layer_call_and_return_conditional_losses_14105215inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_re_lu_46_layer_call_fn_14105220inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_re_lu_46_layer_call_and_return_conditional_losses_14105225inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dense_127_layer_call_fn_14105235inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dense_127_layer_call_and_return_conditional_losses_14105245inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_14105254"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_14105263"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_14105272"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
(:&	�*2Adam/dense_124/kernel/m
!:2Adam/dense_124/bias/m
/:-2#Adam/batch_normalization_86/gamma/m
.:,2"Adam/batch_normalization_86/beta/m
':%@2Adam/dense_125/kernel/m
!:@2Adam/dense_125/bias/m
/:-@2#Adam/batch_normalization_87/gamma/m
.:,@2"Adam/batch_normalization_87/beta/m
':%@@2Adam/dense_126/kernel/m
!:@2Adam/dense_126/bias/m
/:-@2#Adam/batch_normalization_88/gamma/m
.:,@2"Adam/batch_normalization_88/beta/m
':%@2Adam/dense_127/kernel/m
!:2Adam/dense_127/bias/m
(:&	�*2Adam/dense_124/kernel/v
!:2Adam/dense_124/bias/v
/:-2#Adam/batch_normalization_86/gamma/v
.:,2"Adam/batch_normalization_86/beta/v
':%@2Adam/dense_125/kernel/v
!:@2Adam/dense_125/bias/v
/:-@2#Adam/batch_normalization_87/gamma/v
.:,@2"Adam/batch_normalization_87/beta/v
':%@@2Adam/dense_126/kernel/v
!:@2Adam/dense_126/bias/v
/:-@2#Adam/batch_normalization_88/gamma/v
.:,@2"Adam/batch_normalization_88/beta/v
':%@2Adam/dense_127/kernel/v
!:2Adam/dense_127/bias/v�
#__inference__wrapped_model_14102496�#$/,.-CDOLNM\]hegfuv6�3
,�)
'�$
image���������<Z
� "5�2
0
	dense_127#� 
	dense_127����������
T__inference_batch_normalization_86_layer_call_and_return_conditional_losses_14104821b/,.-3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
T__inference_batch_normalization_86_layer_call_and_return_conditional_losses_14104855b./,-3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
9__inference_batch_normalization_86_layer_call_fn_14104767U/,.-3�0
)�&
 �
inputs���������
p 
� "�����������
9__inference_batch_normalization_86_layer_call_fn_14104801U./,-3�0
)�&
 �
inputs���������
p
� "�����������
T__inference_batch_normalization_87_layer_call_and_return_conditional_losses_14105035bOLNM3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
T__inference_batch_normalization_87_layer_call_and_return_conditional_losses_14105069bNOLM3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
9__inference_batch_normalization_87_layer_call_fn_14104981UOLNM3�0
)�&
 �
inputs���������@
p 
� "����������@�
9__inference_batch_normalization_87_layer_call_fn_14105015UNOLM3�0
)�&
 �
inputs���������@
p
� "����������@�
T__inference_batch_normalization_88_layer_call_and_return_conditional_losses_14105181bhegf3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
T__inference_batch_normalization_88_layer_call_and_return_conditional_losses_14105215bghef3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
9__inference_batch_normalization_88_layer_call_fn_14105127Uhegf3�0
)�&
 �
inputs���������@
p 
� "����������@�
9__inference_batch_normalization_88_layer_call_fn_14105161Ughef3�0
)�&
 �
inputs���������@
p
� "����������@�
G__inference_dense_124_layer_call_and_return_conditional_losses_14104747]#$0�-
&�#
!�
inputs����������*
� "%�"
�
0���������
� �
,__inference_dense_124_layer_call_fn_14104733P#$0�-
&�#
!�
inputs����������*
� "�����������
G__inference_dense_125_layer_call_and_return_conditional_losses_14104961\CD/�,
%�"
 �
inputs���������
� "%�"
�
0���������@
� 
,__inference_dense_125_layer_call_fn_14104947OCD/�,
%�"
 �
inputs���������
� "����������@�
G__inference_dense_126_layer_call_and_return_conditional_losses_14105107\\]/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� 
,__inference_dense_126_layer_call_fn_14105093O\]/�,
%�"
 �
inputs���������@
� "����������@�
G__inference_dense_127_layer_call_and_return_conditional_losses_14105245\uv/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� 
,__inference_dense_127_layer_call_fn_14105235Ouv/�,
%�"
 �
inputs���������@
� "�����������
H__inference_dropout_10_layer_call_and_return_conditional_losses_14104904\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
H__inference_dropout_10_layer_call_and_return_conditional_losses_14104916\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
H__inference_dropout_10_layer_call_and_return_conditional_losses_14104921\3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
H__inference_dropout_10_layer_call_and_return_conditional_losses_14104933\3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
-__inference_dropout_10_layer_call_fn_14104870O3�0
)�&
 �
inputs���������@
p 
� "����������@�
-__inference_dropout_10_layer_call_fn_14104882O3�0
)�&
 �
inputs���������@
p
� "����������@�
-__inference_dropout_10_layer_call_fn_14104887O3�0
)�&
 �
inputs���������
p 
� "�����������
-__inference_dropout_10_layer_call_fn_14104899O3�0
)�&
 �
inputs���������
p
� "�����������
H__inference_flatten_39_layer_call_and_return_conditional_losses_14104719a7�4
-�*
(�%
inputs���������<Z
� "&�#
�
0����������*
� �
-__inference_flatten_39_layer_call_fn_14104713T7�4
-�*
(�%
inputs���������<Z
� "�����������*=
__inference_loss_fn_0_14105254#�

� 
� "� =
__inference_loss_fn_1_14105263C�

� 
� "� =
__inference_loss_fn_2_14105272\�

� 
� "� �
F__inference_model_16_layer_call_and_return_conditional_losses_14103977}#$/,.-CDOLNM\]hegfuv>�;
4�1
'�$
image���������<Z
p 

 
� "%�"
�
0���������
� �
F__inference_model_16_layer_call_and_return_conditional_losses_14104140}#$./,-CDNOLM\]ghefuv>�;
4�1
'�$
image���������<Z
p

 
� "%�"
�
0���������
� �
F__inference_model_16_layer_call_and_return_conditional_losses_14104556~#$/,.-CDOLNM\]hegfuv?�<
5�2
(�%
inputs���������<Z
p 

 
� "%�"
�
0���������
� �
F__inference_model_16_layer_call_and_return_conditional_losses_14104707~#$./,-CDNOLM\]ghefuv?�<
5�2
(�%
inputs���������<Z
p

 
� "%�"
�
0���������
� �
+__inference_model_16_layer_call_fn_14102934p#$/,.-CDOLNM\]hegfuv>�;
4�1
'�$
image���������<Z
p 

 
� "�����������
+__inference_model_16_layer_call_fn_14103870p#$./,-CDNOLM\]ghefuv>�;
4�1
'�$
image���������<Z
p

 
� "�����������
+__inference_model_16_layer_call_fn_14104310q#$/,.-CDOLNM\]hegfuv?�<
5�2
(�%
inputs���������<Z
p 

 
� "�����������
+__inference_model_16_layer_call_fn_14104461q#$./,-CDNOLM\]ghefuv?�<
5�2
(�%
inputs���������<Z
p

 
� "�����������
F__inference_re_lu_44_layer_call_and_return_conditional_losses_14104865X/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� z
+__inference_re_lu_44_layer_call_fn_14104860K/�,
%�"
 �
inputs���������
� "�����������
F__inference_re_lu_45_layer_call_and_return_conditional_losses_14105079X/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� z
+__inference_re_lu_45_layer_call_fn_14105074K/�,
%�"
 �
inputs���������@
� "����������@�
F__inference_re_lu_46_layer_call_and_return_conditional_losses_14105225X/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� z
+__inference_re_lu_46_layer_call_fn_14105220K/�,
%�"
 �
inputs���������@
� "����������@�
&__inference_signature_wrapper_14104203�#$/,.-CDOLNM\]hegfuv?�<
� 
5�2
0
image'�$
image���������<Z"5�2
0
	dense_127#� 
	dense_127���������