��
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
 �"serve*2.10.12v2.10.0-76-gfdfc646704c8��
�
Adam/dense_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_33/bias/v
y
(Adam/dense_33/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/dense_33/kernel/v
�
*Adam/dense_33/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/v*
_output_shapes
:	�*
dtype0
�
"Adam/batch_normalization_22/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_22/beta/v
�
6Adam/batch_normalization_22/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_22/beta/v*
_output_shapes	
:�*
dtype0
�
#Adam/batch_normalization_22/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/batch_normalization_22/gamma/v
�
7Adam/batch_normalization_22/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_22/gamma/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_32/bias/v
z
(Adam/dense_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/dense_32/kernel/v
�
*Adam/dense_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/v*
_output_shapes
:	�*
dtype0
�
"Adam/batch_normalization_21/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_21/beta/v
�
6Adam/batch_normalization_21/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_21/beta/v*
_output_shapes
:*
dtype0
�
#Adam/batch_normalization_21/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_21/gamma/v
�
7Adam/batch_normalization_21/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_21/gamma/v*
_output_shapes
:*
dtype0
�
Adam/dense_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_31/bias/v
y
(Adam/dense_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�**'
shared_nameAdam/dense_31/kernel/v
�
*Adam/dense_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/v*
_output_shapes
:	�**
dtype0
�
Adam/dense_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_33/bias/m
y
(Adam/dense_33/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/dense_33/kernel/m
�
*Adam/dense_33/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/m*
_output_shapes
:	�*
dtype0
�
"Adam/batch_normalization_22/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_22/beta/m
�
6Adam/batch_normalization_22/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_22/beta/m*
_output_shapes	
:�*
dtype0
�
#Adam/batch_normalization_22/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#Adam/batch_normalization_22/gamma/m
�
7Adam/batch_normalization_22/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_22/gamma/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_32/bias/m
z
(Adam/dense_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/dense_32/kernel/m
�
*Adam/dense_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/m*
_output_shapes
:	�*
dtype0
�
"Adam/batch_normalization_21/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_21/beta/m
�
6Adam/batch_normalization_21/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_21/beta/m*
_output_shapes
:*
dtype0
�
#Adam/batch_normalization_21/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_21/gamma/m
�
7Adam/batch_normalization_21/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_21/gamma/m*
_output_shapes
:*
dtype0
�
Adam/dense_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_31/bias/m
y
(Adam/dense_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�**'
shared_nameAdam/dense_31/kernel/m
�
*Adam/dense_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/m*
_output_shapes
:	�**
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
r
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_33/bias
k
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
_output_shapes
:*
dtype0
{
dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_33/kernel
t
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel*
_output_shapes
:	�*
dtype0
�
&batch_normalization_22/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_22/moving_variance
�
:batch_normalization_22/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_22/moving_variance*
_output_shapes	
:�*
dtype0
�
"batch_normalization_22/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_22/moving_mean
�
6batch_normalization_22/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_22/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_22/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_22/beta
�
/batch_normalization_22/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_22/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_22/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_22/gamma
�
0batch_normalization_22/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_22/gamma*
_output_shapes	
:�*
dtype0
s
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_32/bias
l
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes	
:�*
dtype0
{
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_32/kernel
t
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel*
_output_shapes
:	�*
dtype0
�
&batch_normalization_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_21/moving_variance
�
:batch_normalization_21/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_21/moving_variance*
_output_shapes
:*
dtype0
�
"batch_normalization_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_21/moving_mean
�
6batch_normalization_21/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_21/moving_mean*
_output_shapes
:*
dtype0
�
batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_21/beta
�
/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_21/beta*
_output_shapes
:*
dtype0
�
batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_21/gamma
�
0batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_21/gamma*
_output_shapes
:*
dtype0
r
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_31/bias
k
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes
:*
dtype0
{
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�** 
shared_namedense_31/kernel
t
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel*
_output_shapes
:	�**
dtype0
�
serving_default_imagePlaceholder*/
_output_shapes
:���������<Z*
dtype0*$
shape:���������<Z
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_imagedense_31/kerneldense_31/bias&batch_normalization_21/moving_variancebatch_normalization_21/gamma"batch_normalization_21/moving_meanbatch_normalization_21/betadense_32/kerneldense_32/bias&batch_normalization_22/moving_variancebatch_normalization_22/gamma"batch_normalization_22/moving_meanbatch_normalization_22/betadense_33/kerneldense_33/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� */
f*R(
&__inference_signature_wrapper_20225741

NoOpNoOp
�Z
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�Z
value�ZB�Z B�Z
�
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
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

 kernel
!bias*
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(axis
	)gamma
*beta
+moving_mean
,moving_variance*
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses* 
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9_random_generator* 
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias*
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Haxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance*
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses* 
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias*
j
 0
!1
)2
*3
+4
,5
@6
A7
I8
J9
K10
L11
Y12
Z13*
J
 0
!1
)2
*3
@4
A5
I6
J7
Y8
Z9*

[0
\1* 
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
btrace_0
ctrace_1
dtrace_2
etrace_3* 
6
ftrace_0
gtrace_1
htrace_2
itrace_3* 
* 
�
jiter

kbeta_1

lbeta_2
	mdecay m�!m�)m�*m�@m�Am�Im�Jm�Ym�Zm� v�!v�)v�*v�@v�Av�Iv�Jv�Yv�Zv�*

nserving_default* 
* 
* 
* 
�
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

ttrace_0* 

utrace_0* 

 0
!1*

 0
!1*
	
[0* 
�
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

{trace_0* 

|trace_0* 
_Y
VARIABLE_VALUEdense_31/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_31/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
)0
*1
+2
,3*

)0
*1*
* 
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_21/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_21/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_21/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_21/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 
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
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

@0
A1*

@0
A1*
	
\0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_32/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_32/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
I0
J1
K2
L3*

I0
J1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_22/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_22/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_22/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_22/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

Y0
Z1*

Y0
Z1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_33/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_33/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 
 
+0
,1
K2
L3*
J
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
9*

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
	
[0* 
* 
* 
* 

+0
,1*
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
	
\0* 
* 
* 
* 

K0
L1*
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
�|
VARIABLE_VALUEAdam/dense_31/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_31/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_21/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_21/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_32/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_32/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_22/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_22/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_33/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_33/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_31/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_31/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_21/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_21/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_32/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_32/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/batch_normalization_22/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_22/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_33/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_33/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
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
�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*�
value�B�)B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOp0batch_normalization_21/gamma/Read/ReadVariableOp/batch_normalization_21/beta/Read/ReadVariableOp6batch_normalization_21/moving_mean/Read/ReadVariableOp:batch_normalization_21/moving_variance/Read/ReadVariableOp#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp0batch_normalization_22/gamma/Read/ReadVariableOp/batch_normalization_22/beta/Read/ReadVariableOp6batch_normalization_22/moving_mean/Read/ReadVariableOp:batch_normalization_22/moving_variance/Read/ReadVariableOp#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_31/kernel/m/Read/ReadVariableOp(Adam/dense_31/bias/m/Read/ReadVariableOp7Adam/batch_normalization_21/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_21/beta/m/Read/ReadVariableOp*Adam/dense_32/kernel/m/Read/ReadVariableOp(Adam/dense_32/bias/m/Read/ReadVariableOp7Adam/batch_normalization_22/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_22/beta/m/Read/ReadVariableOp*Adam/dense_33/kernel/m/Read/ReadVariableOp(Adam/dense_33/bias/m/Read/ReadVariableOp*Adam/dense_31/kernel/v/Read/ReadVariableOp(Adam/dense_31/bias/v/Read/ReadVariableOp7Adam/batch_normalization_21/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_21/beta/v/Read/ReadVariableOp*Adam/dense_32/kernel/v/Read/ReadVariableOp(Adam/dense_32/bias/v/Read/ReadVariableOp7Adam/batch_normalization_22/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_22/beta/v/Read/ReadVariableOp*Adam/dense_33/kernel/v/Read/ReadVariableOp(Adam/dense_33/bias/v/Read/ReadVariableOpConst"/device:CPU:0*7
dtypes-
+2)	
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
�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*�
value�B�)B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::*7
dtypes-
+2)	
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOpAssignVariableOpdense_31/kernel
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_1AssignVariableOpdense_31/bias
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_2AssignVariableOpbatch_normalization_21/gamma
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
k
AssignVariableOp_3AssignVariableOpbatch_normalization_21/beta
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
r
AssignVariableOp_4AssignVariableOp"batch_normalization_21/moving_mean
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
v
AssignVariableOp_5AssignVariableOp&batch_normalization_21/moving_variance
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_6AssignVariableOpdense_32/kernel
Identity_7"/device:CPU:0*
dtype0
U

Identity_8IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_7AssignVariableOpdense_32/bias
Identity_8"/device:CPU:0*
dtype0
U

Identity_9IdentityRestoreV2:8"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_8AssignVariableOpbatch_normalization_22/gamma
Identity_9"/device:CPU:0*
dtype0
V
Identity_10IdentityRestoreV2:9"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_9AssignVariableOpbatch_normalization_22/betaIdentity_10"/device:CPU:0*
dtype0
W
Identity_11IdentityRestoreV2:10"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_10AssignVariableOp"batch_normalization_22/moving_meanIdentity_11"/device:CPU:0*
dtype0
W
Identity_12IdentityRestoreV2:11"/device:CPU:0*
T0*
_output_shapes
:
x
AssignVariableOp_11AssignVariableOp&batch_normalization_22/moving_varianceIdentity_12"/device:CPU:0*
dtype0
W
Identity_13IdentityRestoreV2:12"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_12AssignVariableOpdense_33/kernelIdentity_13"/device:CPU:0*
dtype0
W
Identity_14IdentityRestoreV2:13"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_13AssignVariableOpdense_33/biasIdentity_14"/device:CPU:0*
dtype0
W
Identity_15IdentityRestoreV2:14"/device:CPU:0*
T0	*
_output_shapes
:
[
AssignVariableOp_14AssignVariableOp	Adam/iterIdentity_15"/device:CPU:0*
dtype0	
W
Identity_16IdentityRestoreV2:15"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_15AssignVariableOpAdam/beta_1Identity_16"/device:CPU:0*
dtype0
W
Identity_17IdentityRestoreV2:16"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_16AssignVariableOpAdam/beta_2Identity_17"/device:CPU:0*
dtype0
W
Identity_18IdentityRestoreV2:17"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_17AssignVariableOp
Adam/decayIdentity_18"/device:CPU:0*
dtype0
W
Identity_19IdentityRestoreV2:18"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_18AssignVariableOptotalIdentity_19"/device:CPU:0*
dtype0
W
Identity_20IdentityRestoreV2:19"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_19AssignVariableOpcountIdentity_20"/device:CPU:0*
dtype0
W
Identity_21IdentityRestoreV2:20"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_20AssignVariableOpAdam/dense_31/kernel/mIdentity_21"/device:CPU:0*
dtype0
W
Identity_22IdentityRestoreV2:21"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_21AssignVariableOpAdam/dense_31/bias/mIdentity_22"/device:CPU:0*
dtype0
W
Identity_23IdentityRestoreV2:22"/device:CPU:0*
T0*
_output_shapes
:
u
AssignVariableOp_22AssignVariableOp#Adam/batch_normalization_21/gamma/mIdentity_23"/device:CPU:0*
dtype0
W
Identity_24IdentityRestoreV2:23"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_23AssignVariableOp"Adam/batch_normalization_21/beta/mIdentity_24"/device:CPU:0*
dtype0
W
Identity_25IdentityRestoreV2:24"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_24AssignVariableOpAdam/dense_32/kernel/mIdentity_25"/device:CPU:0*
dtype0
W
Identity_26IdentityRestoreV2:25"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_25AssignVariableOpAdam/dense_32/bias/mIdentity_26"/device:CPU:0*
dtype0
W
Identity_27IdentityRestoreV2:26"/device:CPU:0*
T0*
_output_shapes
:
u
AssignVariableOp_26AssignVariableOp#Adam/batch_normalization_22/gamma/mIdentity_27"/device:CPU:0*
dtype0
W
Identity_28IdentityRestoreV2:27"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_27AssignVariableOp"Adam/batch_normalization_22/beta/mIdentity_28"/device:CPU:0*
dtype0
W
Identity_29IdentityRestoreV2:28"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_28AssignVariableOpAdam/dense_33/kernel/mIdentity_29"/device:CPU:0*
dtype0
W
Identity_30IdentityRestoreV2:29"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_29AssignVariableOpAdam/dense_33/bias/mIdentity_30"/device:CPU:0*
dtype0
W
Identity_31IdentityRestoreV2:30"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_30AssignVariableOpAdam/dense_31/kernel/vIdentity_31"/device:CPU:0*
dtype0
W
Identity_32IdentityRestoreV2:31"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_31AssignVariableOpAdam/dense_31/bias/vIdentity_32"/device:CPU:0*
dtype0
W
Identity_33IdentityRestoreV2:32"/device:CPU:0*
T0*
_output_shapes
:
u
AssignVariableOp_32AssignVariableOp#Adam/batch_normalization_21/gamma/vIdentity_33"/device:CPU:0*
dtype0
W
Identity_34IdentityRestoreV2:33"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_33AssignVariableOp"Adam/batch_normalization_21/beta/vIdentity_34"/device:CPU:0*
dtype0
W
Identity_35IdentityRestoreV2:34"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_34AssignVariableOpAdam/dense_32/kernel/vIdentity_35"/device:CPU:0*
dtype0
W
Identity_36IdentityRestoreV2:35"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_35AssignVariableOpAdam/dense_32/bias/vIdentity_36"/device:CPU:0*
dtype0
W
Identity_37IdentityRestoreV2:36"/device:CPU:0*
T0*
_output_shapes
:
u
AssignVariableOp_36AssignVariableOp#Adam/batch_normalization_22/gamma/vIdentity_37"/device:CPU:0*
dtype0
W
Identity_38IdentityRestoreV2:37"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_37AssignVariableOp"Adam/batch_normalization_22/beta/vIdentity_38"/device:CPU:0*
dtype0
W
Identity_39IdentityRestoreV2:38"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_38AssignVariableOpAdam/dense_33/kernel/vIdentity_39"/device:CPU:0*
dtype0
W
Identity_40IdentityRestoreV2:39"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_39AssignVariableOpAdam/dense_33/bias/vIdentity_40"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
�
Identity_41Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: ��
�i
�
E__inference_model_9_layer_call_and_return_conditional_losses_20225584	
image:
'dense_31_matmul_readvariableop_resource:	�*6
(dense_31_biasadd_readvariableop_resource:F
8batch_normalization_21_batchnorm_readvariableop_resource:J
<batch_normalization_21_batchnorm_mul_readvariableop_resource:H
:batch_normalization_21_batchnorm_readvariableop_1_resource:H
:batch_normalization_21_batchnorm_readvariableop_2_resource::
'dense_32_matmul_readvariableop_resource:	�7
(dense_32_biasadd_readvariableop_resource:	�G
8batch_normalization_22_batchnorm_readvariableop_resource:	�K
<batch_normalization_22_batchnorm_mul_readvariableop_resource:	�I
:batch_normalization_22_batchnorm_readvariableop_1_resource:	�I
:batch_normalization_22_batchnorm_readvariableop_2_resource:	�:
'dense_33_matmul_readvariableop_resource:	�6
(dense_33_biasadd_readvariableop_resource:
identity��/batch_normalization_21/batchnorm/ReadVariableOp�1batch_normalization_21/batchnorm/ReadVariableOp_1�1batch_normalization_21/batchnorm/ReadVariableOp_2�3batch_normalization_21/batchnorm/mul/ReadVariableOp�/batch_normalization_22/batchnorm/ReadVariableOp�1batch_normalization_22/batchnorm/ReadVariableOp_1�1batch_normalization_22/batchnorm/ReadVariableOp_2�3batch_normalization_22/batchnorm/mul/ReadVariableOp�dense_31/BiasAdd/ReadVariableOp�dense_31/MatMul/ReadVariableOp�:dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp�1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp�dense_32/BiasAdd/ReadVariableOp�dense_32/MatMul/ReadVariableOp�:dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp�1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp�dense_33/BiasAdd/ReadVariableOp�dense_33/MatMul/ReadVariableOpa
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  r
flatten_10/ReshapeReshapeimageflatten_10/Const:output:0*
T0*(
_output_shapes
:����������*�
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
dense_31/MatMulMatMulflatten_10/Reshape:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
:dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
+dense_31/dense_31/kernel/Regularizer/L2LossL2LossBdense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: o
*dense_31/dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
(dense_31/dense_31/kernel/Regularizer/mulMul3dense_31/dense_31/kernel/Regularizer/mul/x:output:04dense_31/dense_31/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_21/batchnorm/addAddV27batch_normalization_21/batchnorm/ReadVariableOp:value:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:0;batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_21/batchnorm/mul_1Muldense_31/BiasAdd:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
1batch_normalization_21/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_21_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
&batch_normalization_21/batchnorm/mul_2Mul9batch_normalization_21/batchnorm/ReadVariableOp_1:value:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_21/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_21_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
$batch_normalization_21/batchnorm/subSub9batch_normalization_21/batchnorm/ReadVariableOp_2:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������s
re_lu_18/ReluRelu*batch_normalization_21/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������m
dropout_9/IdentityIdentityre_lu_18/Relu:activations:0*
T0*'
_output_shapes
:����������
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_32/MatMulMatMuldropout_9/Identity:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
+dense_32/dense_32/kernel/Regularizer/L2LossL2LossBdense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: o
*dense_32/dense_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
(dense_32/dense_32/kernel/Regularizer/mulMul3dense_32/dense_32/kernel/Regularizer/mul/x:output:04dense_32/dense_32/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_22/batchnorm/addAddV27batch_normalization_22/batchnorm/ReadVariableOp:value:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:��
3batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:0;batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_22/batchnorm/mul_1Muldense_32/BiasAdd:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
1batch_normalization_22/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_22_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_22/batchnorm/mul_2Mul9batch_normalization_22/batchnorm/ReadVariableOp_1:value:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
1batch_normalization_22/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_22_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_22/batchnorm/subSub9batch_normalization_22/batchnorm/ReadVariableOp_2:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������t
re_lu_19/ReluRelu*batch_normalization_22/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_33/MatMulMatMulre_lu_19/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
"dense_31/kernel/Regularizer/L2LossL2Loss9dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0+dense_31/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_32/kernel/Regularizer/L2LossL2Loss9dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_32/kernel/Regularizer/mulMul*dense_32/kernel/Regularizer/mul/x:output:0+dense_32/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_33/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp0^batch_normalization_21/batchnorm/ReadVariableOp2^batch_normalization_21/batchnorm/ReadVariableOp_12^batch_normalization_21/batchnorm/ReadVariableOp_24^batch_normalization_21/batchnorm/mul/ReadVariableOp0^batch_normalization_22/batchnorm/ReadVariableOp2^batch_normalization_22/batchnorm/ReadVariableOp_12^batch_normalization_22/batchnorm/ReadVariableOp_24^batch_normalization_22/batchnorm/mul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp;^dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_31/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp;^dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_32/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<Z: : : : : : : : : : : : : : 2b
/batch_normalization_21/batchnorm/ReadVariableOp/batch_normalization_21/batchnorm/ReadVariableOp2f
1batch_normalization_21/batchnorm/ReadVariableOp_11batch_normalization_21/batchnorm/ReadVariableOp_12f
1batch_normalization_21/batchnorm/ReadVariableOp_21batch_normalization_21/batchnorm/ReadVariableOp_22j
3batch_normalization_21/batchnorm/mul/ReadVariableOp3batch_normalization_21/batchnorm/mul/ReadVariableOp2b
/batch_normalization_22/batchnorm/ReadVariableOp/batch_normalization_22/batchnorm/ReadVariableOp2f
1batch_normalization_22/batchnorm/ReadVariableOp_11batch_normalization_22/batchnorm/ReadVariableOp_12f
1batch_normalization_22/batchnorm/ReadVariableOp_21batch_normalization_22/batchnorm/ReadVariableOp_22j
3batch_normalization_22/batchnorm/mul/ReadVariableOp3batch_normalization_22/batchnorm/mul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2x
:dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2x
:dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp:V R
/
_output_shapes
:���������<Z

_user_specified_nameimage
�
�
&__inference_signature_wrapper_20225741	
image
unknown:	�*
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallimageunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *,
f'R%
#__inference__wrapped_model_20224576o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<Z: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:���������<Z

_user_specified_nameimage
�\
�
*__inference_model_9_layer_call_fn_20225816

inputs:
'dense_31_matmul_readvariableop_resource:	�*6
(dense_31_biasadd_readvariableop_resource:F
8batch_normalization_21_batchnorm_readvariableop_resource:J
<batch_normalization_21_batchnorm_mul_readvariableop_resource:H
:batch_normalization_21_batchnorm_readvariableop_1_resource:H
:batch_normalization_21_batchnorm_readvariableop_2_resource::
'dense_32_matmul_readvariableop_resource:	�7
(dense_32_biasadd_readvariableop_resource:	�G
8batch_normalization_22_batchnorm_readvariableop_resource:	�K
<batch_normalization_22_batchnorm_mul_readvariableop_resource:	�I
:batch_normalization_22_batchnorm_readvariableop_1_resource:	�I
:batch_normalization_22_batchnorm_readvariableop_2_resource:	�:
'dense_33_matmul_readvariableop_resource:	�6
(dense_33_biasadd_readvariableop_resource:
identity��/batch_normalization_21/batchnorm/ReadVariableOp�1batch_normalization_21/batchnorm/ReadVariableOp_1�1batch_normalization_21/batchnorm/ReadVariableOp_2�3batch_normalization_21/batchnorm/mul/ReadVariableOp�/batch_normalization_22/batchnorm/ReadVariableOp�1batch_normalization_22/batchnorm/ReadVariableOp_1�1batch_normalization_22/batchnorm/ReadVariableOp_2�3batch_normalization_22/batchnorm/mul/ReadVariableOp�dense_31/BiasAdd/ReadVariableOp�dense_31/MatMul/ReadVariableOp�1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp�dense_32/BiasAdd/ReadVariableOp�dense_32/MatMul/ReadVariableOp�1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp�dense_33/BiasAdd/ReadVariableOp�dense_33/MatMul/ReadVariableOpa
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  s
flatten_10/ReshapeReshapeinputsflatten_10/Const:output:0*
T0*(
_output_shapes
:����������*�
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
dense_31/MatMulMatMulflatten_10/Reshape:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_21/batchnorm/addAddV27batch_normalization_21/batchnorm/ReadVariableOp:value:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:0;batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_21/batchnorm/mul_1Muldense_31/BiasAdd:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
1batch_normalization_21/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_21_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
&batch_normalization_21/batchnorm/mul_2Mul9batch_normalization_21/batchnorm/ReadVariableOp_1:value:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_21/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_21_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
$batch_normalization_21/batchnorm/subSub9batch_normalization_21/batchnorm/ReadVariableOp_2:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������s
re_lu_18/ReluRelu*batch_normalization_21/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������m
dropout_9/IdentityIdentityre_lu_18/Relu:activations:0*
T0*'
_output_shapes
:����������
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_32/MatMulMatMuldropout_9/Identity:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_22/batchnorm/addAddV27batch_normalization_22/batchnorm/ReadVariableOp:value:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:��
3batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:0;batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_22/batchnorm/mul_1Muldense_32/BiasAdd:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
1batch_normalization_22/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_22_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_22/batchnorm/mul_2Mul9batch_normalization_22/batchnorm/ReadVariableOp_1:value:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
1batch_normalization_22/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_22_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_22/batchnorm/subSub9batch_normalization_22/batchnorm/ReadVariableOp_2:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������t
re_lu_19/ReluRelu*batch_normalization_22/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_33/MatMulMatMulre_lu_19/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
"dense_31/kernel/Regularizer/L2LossL2Loss9dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0+dense_31/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_32/kernel/Regularizer/L2LossL2Loss9dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_32/kernel/Regularizer/mulMul*dense_32/kernel/Regularizer/mul/x:output:0+dense_32/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_33/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp0^batch_normalization_21/batchnorm/ReadVariableOp2^batch_normalization_21/batchnorm/ReadVariableOp_12^batch_normalization_21/batchnorm/ReadVariableOp_24^batch_normalization_21/batchnorm/mul/ReadVariableOp0^batch_normalization_22/batchnorm/ReadVariableOp2^batch_normalization_22/batchnorm/ReadVariableOp_12^batch_normalization_22/batchnorm/ReadVariableOp_24^batch_normalization_22/batchnorm/mul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp2^dense_31/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp2^dense_32/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<Z: : : : : : : : : : : : : : 2b
/batch_normalization_21/batchnorm/ReadVariableOp/batch_normalization_21/batchnorm/ReadVariableOp2f
1batch_normalization_21/batchnorm/ReadVariableOp_11batch_normalization_21/batchnorm/ReadVariableOp_12f
1batch_normalization_21/batchnorm/ReadVariableOp_21batch_normalization_21/batchnorm/ReadVariableOp_22j
3batch_normalization_21/batchnorm/mul/ReadVariableOp3batch_normalization_21/batchnorm/mul/ReadVariableOp2b
/batch_normalization_22/batchnorm/ReadVariableOp/batch_normalization_22/batchnorm/ReadVariableOp2f
1batch_normalization_22/batchnorm/ReadVariableOp_11batch_normalization_22/batchnorm/ReadVariableOp_12f
1batch_normalization_22/batchnorm/ReadVariableOp_21batch_normalization_22/batchnorm/ReadVariableOp_22j
3batch_normalization_22/batchnorm/mul/ReadVariableOp3batch_normalization_22/batchnorm/mul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2f
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2f
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������<Z
 
_user_specified_nameinputs
�Z
�
#__inference__wrapped_model_20224576	
imageB
/model_9_dense_31_matmul_readvariableop_resource:	�*>
0model_9_dense_31_biasadd_readvariableop_resource:N
@model_9_batch_normalization_21_batchnorm_readvariableop_resource:R
Dmodel_9_batch_normalization_21_batchnorm_mul_readvariableop_resource:P
Bmodel_9_batch_normalization_21_batchnorm_readvariableop_1_resource:P
Bmodel_9_batch_normalization_21_batchnorm_readvariableop_2_resource:B
/model_9_dense_32_matmul_readvariableop_resource:	�?
0model_9_dense_32_biasadd_readvariableop_resource:	�O
@model_9_batch_normalization_22_batchnorm_readvariableop_resource:	�S
Dmodel_9_batch_normalization_22_batchnorm_mul_readvariableop_resource:	�Q
Bmodel_9_batch_normalization_22_batchnorm_readvariableop_1_resource:	�Q
Bmodel_9_batch_normalization_22_batchnorm_readvariableop_2_resource:	�B
/model_9_dense_33_matmul_readvariableop_resource:	�>
0model_9_dense_33_biasadd_readvariableop_resource:
identity��7model_9/batch_normalization_21/batchnorm/ReadVariableOp�9model_9/batch_normalization_21/batchnorm/ReadVariableOp_1�9model_9/batch_normalization_21/batchnorm/ReadVariableOp_2�;model_9/batch_normalization_21/batchnorm/mul/ReadVariableOp�7model_9/batch_normalization_22/batchnorm/ReadVariableOp�9model_9/batch_normalization_22/batchnorm/ReadVariableOp_1�9model_9/batch_normalization_22/batchnorm/ReadVariableOp_2�;model_9/batch_normalization_22/batchnorm/mul/ReadVariableOp�'model_9/dense_31/BiasAdd/ReadVariableOp�&model_9/dense_31/MatMul/ReadVariableOp�'model_9/dense_32/BiasAdd/ReadVariableOp�&model_9/dense_32/MatMul/ReadVariableOp�'model_9/dense_33/BiasAdd/ReadVariableOp�&model_9/dense_33/MatMul/ReadVariableOpi
model_9/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  �
model_9/flatten_10/ReshapeReshapeimage!model_9/flatten_10/Const:output:0*
T0*(
_output_shapes
:����������*�
&model_9/dense_31/MatMul/ReadVariableOpReadVariableOp/model_9_dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
model_9/dense_31/MatMulMatMul#model_9/flatten_10/Reshape:output:0.model_9/dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model_9/dense_31/BiasAdd/ReadVariableOpReadVariableOp0model_9_dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_9/dense_31/BiasAddBiasAdd!model_9/dense_31/MatMul:product:0/model_9/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
7model_9/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp@model_9_batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0s
.model_9/batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
,model_9/batch_normalization_21/batchnorm/addAddV2?model_9/batch_normalization_21/batchnorm/ReadVariableOp:value:07model_9/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
.model_9/batch_normalization_21/batchnorm/RsqrtRsqrt0model_9/batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:�
;model_9/batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_9_batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
,model_9/batch_normalization_21/batchnorm/mulMul2model_9/batch_normalization_21/batchnorm/Rsqrt:y:0Cmodel_9/batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
.model_9/batch_normalization_21/batchnorm/mul_1Mul!model_9/dense_31/BiasAdd:output:00model_9/batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
9model_9/batch_normalization_21/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_9_batch_normalization_21_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
.model_9/batch_normalization_21/batchnorm/mul_2MulAmodel_9/batch_normalization_21/batchnorm/ReadVariableOp_1:value:00model_9/batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:�
9model_9/batch_normalization_21/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_9_batch_normalization_21_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
,model_9/batch_normalization_21/batchnorm/subSubAmodel_9/batch_normalization_21/batchnorm/ReadVariableOp_2:value:02model_9/batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
.model_9/batch_normalization_21/batchnorm/add_1AddV22model_9/batch_normalization_21/batchnorm/mul_1:z:00model_9/batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
model_9/re_lu_18/ReluRelu2model_9/batch_normalization_21/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������}
model_9/dropout_9/IdentityIdentity#model_9/re_lu_18/Relu:activations:0*
T0*'
_output_shapes
:����������
&model_9/dense_32/MatMul/ReadVariableOpReadVariableOp/model_9_dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model_9/dense_32/MatMulMatMul#model_9/dropout_9/Identity:output:0.model_9/dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'model_9/dense_32/BiasAdd/ReadVariableOpReadVariableOp0model_9_dense_32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_9/dense_32/BiasAddBiasAdd!model_9/dense_32/MatMul:product:0/model_9/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
7model_9/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp@model_9_batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0s
.model_9/batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
,model_9/batch_normalization_22/batchnorm/addAddV2?model_9/batch_normalization_22/batchnorm/ReadVariableOp:value:07model_9/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
.model_9/batch_normalization_22/batchnorm/RsqrtRsqrt0model_9/batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:��
;model_9/batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_9_batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,model_9/batch_normalization_22/batchnorm/mulMul2model_9/batch_normalization_22/batchnorm/Rsqrt:y:0Cmodel_9/batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
.model_9/batch_normalization_22/batchnorm/mul_1Mul!model_9/dense_32/BiasAdd:output:00model_9/batch_normalization_22/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
9model_9/batch_normalization_22/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_9_batch_normalization_22_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
.model_9/batch_normalization_22/batchnorm/mul_2MulAmodel_9/batch_normalization_22/batchnorm/ReadVariableOp_1:value:00model_9/batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
9model_9/batch_normalization_22/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_9_batch_normalization_22_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
,model_9/batch_normalization_22/batchnorm/subSubAmodel_9/batch_normalization_22/batchnorm/ReadVariableOp_2:value:02model_9/batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
.model_9/batch_normalization_22/batchnorm/add_1AddV22model_9/batch_normalization_22/batchnorm/mul_1:z:00model_9/batch_normalization_22/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
model_9/re_lu_19/ReluRelu2model_9/batch_normalization_22/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
&model_9/dense_33/MatMul/ReadVariableOpReadVariableOp/model_9_dense_33_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model_9/dense_33/MatMulMatMul#model_9/re_lu_19/Relu:activations:0.model_9/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model_9/dense_33/BiasAdd/ReadVariableOpReadVariableOp0model_9_dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_9/dense_33/BiasAddBiasAdd!model_9/dense_33/MatMul:product:0/model_9/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
IdentityIdentity!model_9/dense_33/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp8^model_9/batch_normalization_21/batchnorm/ReadVariableOp:^model_9/batch_normalization_21/batchnorm/ReadVariableOp_1:^model_9/batch_normalization_21/batchnorm/ReadVariableOp_2<^model_9/batch_normalization_21/batchnorm/mul/ReadVariableOp8^model_9/batch_normalization_22/batchnorm/ReadVariableOp:^model_9/batch_normalization_22/batchnorm/ReadVariableOp_1:^model_9/batch_normalization_22/batchnorm/ReadVariableOp_2<^model_9/batch_normalization_22/batchnorm/mul/ReadVariableOp(^model_9/dense_31/BiasAdd/ReadVariableOp'^model_9/dense_31/MatMul/ReadVariableOp(^model_9/dense_32/BiasAdd/ReadVariableOp'^model_9/dense_32/MatMul/ReadVariableOp(^model_9/dense_33/BiasAdd/ReadVariableOp'^model_9/dense_33/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<Z: : : : : : : : : : : : : : 2r
7model_9/batch_normalization_21/batchnorm/ReadVariableOp7model_9/batch_normalization_21/batchnorm/ReadVariableOp2v
9model_9/batch_normalization_21/batchnorm/ReadVariableOp_19model_9/batch_normalization_21/batchnorm/ReadVariableOp_12v
9model_9/batch_normalization_21/batchnorm/ReadVariableOp_29model_9/batch_normalization_21/batchnorm/ReadVariableOp_22z
;model_9/batch_normalization_21/batchnorm/mul/ReadVariableOp;model_9/batch_normalization_21/batchnorm/mul/ReadVariableOp2r
7model_9/batch_normalization_22/batchnorm/ReadVariableOp7model_9/batch_normalization_22/batchnorm/ReadVariableOp2v
9model_9/batch_normalization_22/batchnorm/ReadVariableOp_19model_9/batch_normalization_22/batchnorm/ReadVariableOp_12v
9model_9/batch_normalization_22/batchnorm/ReadVariableOp_29model_9/batch_normalization_22/batchnorm/ReadVariableOp_22z
;model_9/batch_normalization_22/batchnorm/mul/ReadVariableOp;model_9/batch_normalization_22/batchnorm/mul/ReadVariableOp2R
'model_9/dense_31/BiasAdd/ReadVariableOp'model_9/dense_31/BiasAdd/ReadVariableOp2P
&model_9/dense_31/MatMul/ReadVariableOp&model_9/dense_31/MatMul/ReadVariableOp2R
'model_9/dense_32/BiasAdd/ReadVariableOp'model_9/dense_32/BiasAdd/ReadVariableOp2P
&model_9/dense_32/MatMul/ReadVariableOp&model_9/dense_32/MatMul/ReadVariableOp2R
'model_9/dense_33/BiasAdd/ReadVariableOp'model_9/dense_33/BiasAdd/ReadVariableOp2P
&model_9/dense_33/MatMul/ReadVariableOp&model_9/dense_33/MatMul/ReadVariableOp:V R
/
_output_shapes
:���������<Z

_user_specified_nameimage
�\
�
E__inference_model_9_layer_call_and_return_conditional_losses_20225985

inputs:
'dense_31_matmul_readvariableop_resource:	�*6
(dense_31_biasadd_readvariableop_resource:F
8batch_normalization_21_batchnorm_readvariableop_resource:J
<batch_normalization_21_batchnorm_mul_readvariableop_resource:H
:batch_normalization_21_batchnorm_readvariableop_1_resource:H
:batch_normalization_21_batchnorm_readvariableop_2_resource::
'dense_32_matmul_readvariableop_resource:	�7
(dense_32_biasadd_readvariableop_resource:	�G
8batch_normalization_22_batchnorm_readvariableop_resource:	�K
<batch_normalization_22_batchnorm_mul_readvariableop_resource:	�I
:batch_normalization_22_batchnorm_readvariableop_1_resource:	�I
:batch_normalization_22_batchnorm_readvariableop_2_resource:	�:
'dense_33_matmul_readvariableop_resource:	�6
(dense_33_biasadd_readvariableop_resource:
identity��/batch_normalization_21/batchnorm/ReadVariableOp�1batch_normalization_21/batchnorm/ReadVariableOp_1�1batch_normalization_21/batchnorm/ReadVariableOp_2�3batch_normalization_21/batchnorm/mul/ReadVariableOp�/batch_normalization_22/batchnorm/ReadVariableOp�1batch_normalization_22/batchnorm/ReadVariableOp_1�1batch_normalization_22/batchnorm/ReadVariableOp_2�3batch_normalization_22/batchnorm/mul/ReadVariableOp�dense_31/BiasAdd/ReadVariableOp�dense_31/MatMul/ReadVariableOp�1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp�dense_32/BiasAdd/ReadVariableOp�dense_32/MatMul/ReadVariableOp�1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp�dense_33/BiasAdd/ReadVariableOp�dense_33/MatMul/ReadVariableOpa
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  s
flatten_10/ReshapeReshapeinputsflatten_10/Const:output:0*
T0*(
_output_shapes
:����������*�
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
dense_31/MatMulMatMulflatten_10/Reshape:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_21/batchnorm/addAddV27batch_normalization_21/batchnorm/ReadVariableOp:value:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:0;batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_21/batchnorm/mul_1Muldense_31/BiasAdd:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
1batch_normalization_21/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_21_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
&batch_normalization_21/batchnorm/mul_2Mul9batch_normalization_21/batchnorm/ReadVariableOp_1:value:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_21/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_21_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
$batch_normalization_21/batchnorm/subSub9batch_normalization_21/batchnorm/ReadVariableOp_2:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������s
re_lu_18/ReluRelu*batch_normalization_21/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������m
dropout_9/IdentityIdentityre_lu_18/Relu:activations:0*
T0*'
_output_shapes
:����������
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_32/MatMulMatMuldropout_9/Identity:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_22/batchnorm/addAddV27batch_normalization_22/batchnorm/ReadVariableOp:value:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:��
3batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:0;batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_22/batchnorm/mul_1Muldense_32/BiasAdd:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
1batch_normalization_22/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_22_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_22/batchnorm/mul_2Mul9batch_normalization_22/batchnorm/ReadVariableOp_1:value:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
1batch_normalization_22/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_22_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_22/batchnorm/subSub9batch_normalization_22/batchnorm/ReadVariableOp_2:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������t
re_lu_19/ReluRelu*batch_normalization_22/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_33/MatMulMatMulre_lu_19/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
"dense_31/kernel/Regularizer/L2LossL2Loss9dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0+dense_31/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_32/kernel/Regularizer/L2LossL2Loss9dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_32/kernel/Regularizer/mulMul*dense_32/kernel/Regularizer/mul/x:output:0+dense_32/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_33/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp0^batch_normalization_21/batchnorm/ReadVariableOp2^batch_normalization_21/batchnorm/ReadVariableOp_12^batch_normalization_21/batchnorm/ReadVariableOp_24^batch_normalization_21/batchnorm/mul/ReadVariableOp0^batch_normalization_22/batchnorm/ReadVariableOp2^batch_normalization_22/batchnorm/ReadVariableOp_12^batch_normalization_22/batchnorm/ReadVariableOp_24^batch_normalization_22/batchnorm/mul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp2^dense_31/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp2^dense_32/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<Z: : : : : : : : : : : : : : 2b
/batch_normalization_21/batchnorm/ReadVariableOp/batch_normalization_21/batchnorm/ReadVariableOp2f
1batch_normalization_21/batchnorm/ReadVariableOp_11batch_normalization_21/batchnorm/ReadVariableOp_12f
1batch_normalization_21/batchnorm/ReadVariableOp_21batch_normalization_21/batchnorm/ReadVariableOp_22j
3batch_normalization_21/batchnorm/mul/ReadVariableOp3batch_normalization_21/batchnorm/mul/ReadVariableOp2b
/batch_normalization_22/batchnorm/ReadVariableOp/batch_normalization_22/batchnorm/ReadVariableOp2f
1batch_normalization_22/batchnorm/ReadVariableOp_11batch_normalization_22/batchnorm/ReadVariableOp_12f
1batch_normalization_22/batchnorm/ReadVariableOp_21batch_normalization_22/batchnorm/ReadVariableOp_22j
3batch_normalization_22/batchnorm/mul/ReadVariableOp3batch_normalization_22/batchnorm/mul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2f
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2f
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������<Z
 
_user_specified_nameinputs
�%
�
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_20226415

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
9__inference_batch_normalization_21_layer_call_fn_20226181

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
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
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
+__inference_dense_33_layer_call_fn_20226435

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
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
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_1_20226463M
:dense_32_kernel_regularizer_l2loss_readvariableop_resource:	�
identity��1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp�
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_32_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_32/kernel/Regularizer/L2LossL2Loss9dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_32/kernel/Regularizer/mulMul*dense_32/kernel/Regularizer/mul/x:output:0+dense_32/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_32/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_32/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp
�%
�
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_20226235

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
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
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
ė
�
*__inference_model_9_layer_call_fn_20225918

inputs:
'dense_31_matmul_readvariableop_resource:	�*6
(dense_31_biasadd_readvariableop_resource:L
>batch_normalization_21_assignmovingavg_readvariableop_resource:N
@batch_normalization_21_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_21_batchnorm_mul_readvariableop_resource:F
8batch_normalization_21_batchnorm_readvariableop_resource::
'dense_32_matmul_readvariableop_resource:	�7
(dense_32_biasadd_readvariableop_resource:	�M
>batch_normalization_22_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_22_assignmovingavg_1_readvariableop_resource:	�K
<batch_normalization_22_batchnorm_mul_readvariableop_resource:	�G
8batch_normalization_22_batchnorm_readvariableop_resource:	�:
'dense_33_matmul_readvariableop_resource:	�6
(dense_33_biasadd_readvariableop_resource:
identity��&batch_normalization_21/AssignMovingAvg�5batch_normalization_21/AssignMovingAvg/ReadVariableOp�(batch_normalization_21/AssignMovingAvg_1�7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_21/batchnorm/ReadVariableOp�3batch_normalization_21/batchnorm/mul/ReadVariableOp�&batch_normalization_22/AssignMovingAvg�5batch_normalization_22/AssignMovingAvg/ReadVariableOp�(batch_normalization_22/AssignMovingAvg_1�7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_22/batchnorm/ReadVariableOp�3batch_normalization_22/batchnorm/mul/ReadVariableOp�dense_31/BiasAdd/ReadVariableOp�dense_31/MatMul/ReadVariableOp�1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp�dense_32/BiasAdd/ReadVariableOp�dense_32/MatMul/ReadVariableOp�1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp�dense_33/BiasAdd/ReadVariableOp�dense_33/MatMul/ReadVariableOpa
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  s
flatten_10/ReshapeReshapeinputsflatten_10/Const:output:0*
T0*(
_output_shapes
:����������*�
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
dense_31/MatMulMatMulflatten_10/Reshape:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
5batch_normalization_21/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_21/moments/meanMeandense_31/BiasAdd:output:0>batch_normalization_21/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
+batch_normalization_21/moments/StopGradientStopGradient,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes

:�
0batch_normalization_21/moments/SquaredDifferenceSquaredDifferencedense_31/BiasAdd:output:04batch_normalization_21/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
9batch_normalization_21/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_21/moments/varianceMean4batch_normalization_21/moments/SquaredDifference:z:0Bbatch_normalization_21/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
&batch_normalization_21/moments/SqueezeSqueeze,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
(batch_normalization_21/moments/Squeeze_1Squeeze0batch_normalization_21/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_21/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_21/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_21_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_21/AssignMovingAvg/subSub=batch_normalization_21/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_21/moments/Squeeze:output:0*
T0*
_output_shapes
:�
*batch_normalization_21/AssignMovingAvg/mulMul.batch_normalization_21/AssignMovingAvg/sub:z:05batch_normalization_21/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
&batch_normalization_21/AssignMovingAvgAssignSubVariableOp>batch_normalization_21_assignmovingavg_readvariableop_resource.batch_normalization_21/AssignMovingAvg/mul:z:06^batch_normalization_21/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_21/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_21_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_21/AssignMovingAvg_1/subSub?batch_normalization_21/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_21/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
,batch_normalization_21/AssignMovingAvg_1/mulMul0batch_normalization_21/AssignMovingAvg_1/sub:z:07batch_normalization_21/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_21/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_21_assignmovingavg_1_readvariableop_resource0batch_normalization_21/AssignMovingAvg_1/mul:z:08^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_21/batchnorm/addAddV21batch_normalization_21/moments/Squeeze_1:output:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:0;batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_21/batchnorm/mul_1Muldense_31/BiasAdd:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
&batch_normalization_21/batchnorm/mul_2Mul/batch_normalization_21/moments/Squeeze:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:�
/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_21/batchnorm/subSub7batch_normalization_21/batchnorm/ReadVariableOp:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������s
re_lu_18/ReluRelu*batch_normalization_21/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������\
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_9/dropout/MulMulre_lu_18/Relu:activations:0 dropout_9/dropout/Const:output:0*
T0*'
_output_shapes
:���������b
dropout_9/dropout/ShapeShapere_lu_18/Relu:activations:0*
T0*
_output_shapes
:�
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0e
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*'
_output_shapes
:����������
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_32/MatMulMatMuldropout_9/dropout/Mul_1:z:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
5batch_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_22/moments/meanMeandense_32/BiasAdd:output:0>batch_normalization_22/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_22/moments/StopGradientStopGradient,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_22/moments/SquaredDifferenceSquaredDifferencedense_32/BiasAdd:output:04batch_normalization_22/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
9batch_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_22/moments/varianceMean4batch_normalization_22/moments/SquaredDifference:z:0Bbatch_normalization_22/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_22/moments/SqueezeSqueeze,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_22/moments/Squeeze_1Squeeze0batch_normalization_22/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_22/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_22/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_22_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_22/AssignMovingAvg/subSub=batch_normalization_22/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_22/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_22/AssignMovingAvg/mulMul.batch_normalization_22/AssignMovingAvg/sub:z:05batch_normalization_22/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_22/AssignMovingAvgAssignSubVariableOp>batch_normalization_22_assignmovingavg_readvariableop_resource.batch_normalization_22/AssignMovingAvg/mul:z:06^batch_normalization_22/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_22/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_22_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_22/AssignMovingAvg_1/subSub?batch_normalization_22/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_22/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_22/AssignMovingAvg_1/mulMul0batch_normalization_22/AssignMovingAvg_1/sub:z:07batch_normalization_22/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_22/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_22_assignmovingavg_1_readvariableop_resource0batch_normalization_22/AssignMovingAvg_1/mul:z:08^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_22/batchnorm/addAddV21batch_normalization_22/moments/Squeeze_1:output:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:��
3batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:0;batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_22/batchnorm/mul_1Muldense_32/BiasAdd:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_22/batchnorm/mul_2Mul/batch_normalization_22/moments/Squeeze:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_22/batchnorm/subSub7batch_normalization_22/batchnorm/ReadVariableOp:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������t
re_lu_19/ReluRelu*batch_normalization_22/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_33/MatMulMatMulre_lu_19/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
"dense_31/kernel/Regularizer/L2LossL2Loss9dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0+dense_31/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_32/kernel/Regularizer/L2LossL2Loss9dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_32/kernel/Regularizer/mulMul*dense_32/kernel/Regularizer/mul/x:output:0+dense_32/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_33/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^batch_normalization_21/AssignMovingAvg6^batch_normalization_21/AssignMovingAvg/ReadVariableOp)^batch_normalization_21/AssignMovingAvg_18^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_21/batchnorm/ReadVariableOp4^batch_normalization_21/batchnorm/mul/ReadVariableOp'^batch_normalization_22/AssignMovingAvg6^batch_normalization_22/AssignMovingAvg/ReadVariableOp)^batch_normalization_22/AssignMovingAvg_18^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_22/batchnorm/ReadVariableOp4^batch_normalization_22/batchnorm/mul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp2^dense_31/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp2^dense_32/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<Z: : : : : : : : : : : : : : 2P
&batch_normalization_21/AssignMovingAvg&batch_normalization_21/AssignMovingAvg2n
5batch_normalization_21/AssignMovingAvg/ReadVariableOp5batch_normalization_21/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_21/AssignMovingAvg_1(batch_normalization_21/AssignMovingAvg_12r
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_21/batchnorm/ReadVariableOp/batch_normalization_21/batchnorm/ReadVariableOp2j
3batch_normalization_21/batchnorm/mul/ReadVariableOp3batch_normalization_21/batchnorm/mul/ReadVariableOp2P
&batch_normalization_22/AssignMovingAvg&batch_normalization_22/AssignMovingAvg2n
5batch_normalization_22/AssignMovingAvg/ReadVariableOp5batch_normalization_22/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_22/AssignMovingAvg_1(batch_normalization_22/AssignMovingAvg_12r
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_22/batchnorm/ReadVariableOp/batch_normalization_22/batchnorm/ReadVariableOp2j
3batch_normalization_22/batchnorm/mul/ReadVariableOp3batch_normalization_22/batchnorm/mul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2f
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2f
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������<Z
 
_user_specified_nameinputs
ߗ
�
E__inference_model_9_layer_call_and_return_conditional_losses_20226087

inputs:
'dense_31_matmul_readvariableop_resource:	�*6
(dense_31_biasadd_readvariableop_resource:L
>batch_normalization_21_assignmovingavg_readvariableop_resource:N
@batch_normalization_21_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_21_batchnorm_mul_readvariableop_resource:F
8batch_normalization_21_batchnorm_readvariableop_resource::
'dense_32_matmul_readvariableop_resource:	�7
(dense_32_biasadd_readvariableop_resource:	�M
>batch_normalization_22_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_22_assignmovingavg_1_readvariableop_resource:	�K
<batch_normalization_22_batchnorm_mul_readvariableop_resource:	�G
8batch_normalization_22_batchnorm_readvariableop_resource:	�:
'dense_33_matmul_readvariableop_resource:	�6
(dense_33_biasadd_readvariableop_resource:
identity��&batch_normalization_21/AssignMovingAvg�5batch_normalization_21/AssignMovingAvg/ReadVariableOp�(batch_normalization_21/AssignMovingAvg_1�7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_21/batchnorm/ReadVariableOp�3batch_normalization_21/batchnorm/mul/ReadVariableOp�&batch_normalization_22/AssignMovingAvg�5batch_normalization_22/AssignMovingAvg/ReadVariableOp�(batch_normalization_22/AssignMovingAvg_1�7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_22/batchnorm/ReadVariableOp�3batch_normalization_22/batchnorm/mul/ReadVariableOp�dense_31/BiasAdd/ReadVariableOp�dense_31/MatMul/ReadVariableOp�1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp�dense_32/BiasAdd/ReadVariableOp�dense_32/MatMul/ReadVariableOp�1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp�dense_33/BiasAdd/ReadVariableOp�dense_33/MatMul/ReadVariableOpa
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  s
flatten_10/ReshapeReshapeinputsflatten_10/Const:output:0*
T0*(
_output_shapes
:����������*�
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
dense_31/MatMulMatMulflatten_10/Reshape:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
5batch_normalization_21/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_21/moments/meanMeandense_31/BiasAdd:output:0>batch_normalization_21/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
+batch_normalization_21/moments/StopGradientStopGradient,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes

:�
0batch_normalization_21/moments/SquaredDifferenceSquaredDifferencedense_31/BiasAdd:output:04batch_normalization_21/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
9batch_normalization_21/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_21/moments/varianceMean4batch_normalization_21/moments/SquaredDifference:z:0Bbatch_normalization_21/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
&batch_normalization_21/moments/SqueezeSqueeze,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
(batch_normalization_21/moments/Squeeze_1Squeeze0batch_normalization_21/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_21/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_21/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_21_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_21/AssignMovingAvg/subSub=batch_normalization_21/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_21/moments/Squeeze:output:0*
T0*
_output_shapes
:�
*batch_normalization_21/AssignMovingAvg/mulMul.batch_normalization_21/AssignMovingAvg/sub:z:05batch_normalization_21/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
&batch_normalization_21/AssignMovingAvgAssignSubVariableOp>batch_normalization_21_assignmovingavg_readvariableop_resource.batch_normalization_21/AssignMovingAvg/mul:z:06^batch_normalization_21/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_21/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_21_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_21/AssignMovingAvg_1/subSub?batch_normalization_21/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_21/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
,batch_normalization_21/AssignMovingAvg_1/mulMul0batch_normalization_21/AssignMovingAvg_1/sub:z:07batch_normalization_21/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_21/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_21_assignmovingavg_1_readvariableop_resource0batch_normalization_21/AssignMovingAvg_1/mul:z:08^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_21/batchnorm/addAddV21batch_normalization_21/moments/Squeeze_1:output:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:0;batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_21/batchnorm/mul_1Muldense_31/BiasAdd:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
&batch_normalization_21/batchnorm/mul_2Mul/batch_normalization_21/moments/Squeeze:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:�
/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_21/batchnorm/subSub7batch_normalization_21/batchnorm/ReadVariableOp:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������s
re_lu_18/ReluRelu*batch_normalization_21/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������\
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_9/dropout/MulMulre_lu_18/Relu:activations:0 dropout_9/dropout/Const:output:0*
T0*'
_output_shapes
:���������b
dropout_9/dropout/ShapeShapere_lu_18/Relu:activations:0*
T0*
_output_shapes
:�
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0e
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*'
_output_shapes
:����������
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_32/MatMulMatMuldropout_9/dropout/Mul_1:z:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
5batch_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_22/moments/meanMeandense_32/BiasAdd:output:0>batch_normalization_22/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_22/moments/StopGradientStopGradient,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_22/moments/SquaredDifferenceSquaredDifferencedense_32/BiasAdd:output:04batch_normalization_22/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
9batch_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_22/moments/varianceMean4batch_normalization_22/moments/SquaredDifference:z:0Bbatch_normalization_22/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_22/moments/SqueezeSqueeze,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_22/moments/Squeeze_1Squeeze0batch_normalization_22/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_22/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_22/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_22_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_22/AssignMovingAvg/subSub=batch_normalization_22/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_22/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_22/AssignMovingAvg/mulMul.batch_normalization_22/AssignMovingAvg/sub:z:05batch_normalization_22/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_22/AssignMovingAvgAssignSubVariableOp>batch_normalization_22_assignmovingavg_readvariableop_resource.batch_normalization_22/AssignMovingAvg/mul:z:06^batch_normalization_22/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_22/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_22_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_22/AssignMovingAvg_1/subSub?batch_normalization_22/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_22/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_22/AssignMovingAvg_1/mulMul0batch_normalization_22/AssignMovingAvg_1/sub:z:07batch_normalization_22/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_22/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_22_assignmovingavg_1_readvariableop_resource0batch_normalization_22/AssignMovingAvg_1/mul:z:08^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_22/batchnorm/addAddV21batch_normalization_22/moments/Squeeze_1:output:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:��
3batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:0;batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_22/batchnorm/mul_1Muldense_32/BiasAdd:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_22/batchnorm/mul_2Mul/batch_normalization_22/moments/Squeeze:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_22/batchnorm/subSub7batch_normalization_22/batchnorm/ReadVariableOp:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������t
re_lu_19/ReluRelu*batch_normalization_22/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_33/MatMulMatMulre_lu_19/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
"dense_31/kernel/Regularizer/L2LossL2Loss9dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0+dense_31/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_32/kernel/Regularizer/L2LossL2Loss9dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_32/kernel/Regularizer/mulMul*dense_32/kernel/Regularizer/mul/x:output:0+dense_32/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_33/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^batch_normalization_21/AssignMovingAvg6^batch_normalization_21/AssignMovingAvg/ReadVariableOp)^batch_normalization_21/AssignMovingAvg_18^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_21/batchnorm/ReadVariableOp4^batch_normalization_21/batchnorm/mul/ReadVariableOp'^batch_normalization_22/AssignMovingAvg6^batch_normalization_22/AssignMovingAvg/ReadVariableOp)^batch_normalization_22/AssignMovingAvg_18^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_22/batchnorm/ReadVariableOp4^batch_normalization_22/batchnorm/mul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp2^dense_31/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp2^dense_32/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<Z: : : : : : : : : : : : : : 2P
&batch_normalization_21/AssignMovingAvg&batch_normalization_21/AssignMovingAvg2n
5batch_normalization_21/AssignMovingAvg/ReadVariableOp5batch_normalization_21/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_21/AssignMovingAvg_1(batch_normalization_21/AssignMovingAvg_12r
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_21/batchnorm/ReadVariableOp/batch_normalization_21/batchnorm/ReadVariableOp2j
3batch_normalization_21/batchnorm/mul/ReadVariableOp3batch_normalization_21/batchnorm/mul/ReadVariableOp2P
&batch_normalization_22/AssignMovingAvg&batch_normalization_22/AssignMovingAvg2n
5batch_normalization_22/AssignMovingAvg/ReadVariableOp5batch_normalization_22/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_22/AssignMovingAvg_1(batch_normalization_22/AssignMovingAvg_12r
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_22/batchnorm/ReadVariableOp/batch_normalization_22/batchnorm/ReadVariableOp2j
3batch_normalization_22/batchnorm/mul/ReadVariableOp3batch_normalization_22/batchnorm/mul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2f
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2f
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������<Z
 
_user_specified_nameinputs
�	
f
G__inference_dropout_9_layer_call_and_return_conditional_losses_20226279

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�i
�
*__inference_model_9_layer_call_fn_20224872	
image:
'dense_31_matmul_readvariableop_resource:	�*6
(dense_31_biasadd_readvariableop_resource:F
8batch_normalization_21_batchnorm_readvariableop_resource:J
<batch_normalization_21_batchnorm_mul_readvariableop_resource:H
:batch_normalization_21_batchnorm_readvariableop_1_resource:H
:batch_normalization_21_batchnorm_readvariableop_2_resource::
'dense_32_matmul_readvariableop_resource:	�7
(dense_32_biasadd_readvariableop_resource:	�G
8batch_normalization_22_batchnorm_readvariableop_resource:	�K
<batch_normalization_22_batchnorm_mul_readvariableop_resource:	�I
:batch_normalization_22_batchnorm_readvariableop_1_resource:	�I
:batch_normalization_22_batchnorm_readvariableop_2_resource:	�:
'dense_33_matmul_readvariableop_resource:	�6
(dense_33_biasadd_readvariableop_resource:
identity��/batch_normalization_21/batchnorm/ReadVariableOp�1batch_normalization_21/batchnorm/ReadVariableOp_1�1batch_normalization_21/batchnorm/ReadVariableOp_2�3batch_normalization_21/batchnorm/mul/ReadVariableOp�/batch_normalization_22/batchnorm/ReadVariableOp�1batch_normalization_22/batchnorm/ReadVariableOp_1�1batch_normalization_22/batchnorm/ReadVariableOp_2�3batch_normalization_22/batchnorm/mul/ReadVariableOp�dense_31/BiasAdd/ReadVariableOp�dense_31/MatMul/ReadVariableOp�:dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp�1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp�dense_32/BiasAdd/ReadVariableOp�dense_32/MatMul/ReadVariableOp�:dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp�1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp�dense_33/BiasAdd/ReadVariableOp�dense_33/MatMul/ReadVariableOpa
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  r
flatten_10/ReshapeReshapeimageflatten_10/Const:output:0*
T0*(
_output_shapes
:����������*�
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
dense_31/MatMulMatMulflatten_10/Reshape:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
:dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
+dense_31/dense_31/kernel/Regularizer/L2LossL2LossBdense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: o
*dense_31/dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
(dense_31/dense_31/kernel/Regularizer/mulMul3dense_31/dense_31/kernel/Regularizer/mul/x:output:04dense_31/dense_31/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_21/batchnorm/addAddV27batch_normalization_21/batchnorm/ReadVariableOp:value:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:0;batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_21/batchnorm/mul_1Muldense_31/BiasAdd:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
1batch_normalization_21/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_21_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
&batch_normalization_21/batchnorm/mul_2Mul9batch_normalization_21/batchnorm/ReadVariableOp_1:value:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:�
1batch_normalization_21/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_21_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
$batch_normalization_21/batchnorm/subSub9batch_normalization_21/batchnorm/ReadVariableOp_2:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������s
re_lu_18/ReluRelu*batch_normalization_21/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������m
dropout_9/IdentityIdentityre_lu_18/Relu:activations:0*
T0*'
_output_shapes
:����������
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_32/MatMulMatMuldropout_9/Identity:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
+dense_32/dense_32/kernel/Regularizer/L2LossL2LossBdense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: o
*dense_32/dense_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
(dense_32/dense_32/kernel/Regularizer/mulMul3dense_32/dense_32/kernel/Regularizer/mul/x:output:04dense_32/dense_32/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0k
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_22/batchnorm/addAddV27batch_normalization_22/batchnorm/ReadVariableOp:value:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:��
3batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:0;batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_22/batchnorm/mul_1Muldense_32/BiasAdd:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
1batch_normalization_22/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_22_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_22/batchnorm/mul_2Mul9batch_normalization_22/batchnorm/ReadVariableOp_1:value:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
1batch_normalization_22/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_22_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_22/batchnorm/subSub9batch_normalization_22/batchnorm/ReadVariableOp_2:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������t
re_lu_19/ReluRelu*batch_normalization_22/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_33/MatMulMatMulre_lu_19/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
"dense_31/kernel/Regularizer/L2LossL2Loss9dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0+dense_31/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_32/kernel/Regularizer/L2LossL2Loss9dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_32/kernel/Regularizer/mulMul*dense_32/kernel/Regularizer/mul/x:output:0+dense_32/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_33/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp0^batch_normalization_21/batchnorm/ReadVariableOp2^batch_normalization_21/batchnorm/ReadVariableOp_12^batch_normalization_21/batchnorm/ReadVariableOp_24^batch_normalization_21/batchnorm/mul/ReadVariableOp0^batch_normalization_22/batchnorm/ReadVariableOp2^batch_normalization_22/batchnorm/ReadVariableOp_12^batch_normalization_22/batchnorm/ReadVariableOp_24^batch_normalization_22/batchnorm/mul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp;^dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_31/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp;^dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_32/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<Z: : : : : : : : : : : : : : 2b
/batch_normalization_21/batchnorm/ReadVariableOp/batch_normalization_21/batchnorm/ReadVariableOp2f
1batch_normalization_21/batchnorm/ReadVariableOp_11batch_normalization_21/batchnorm/ReadVariableOp_12f
1batch_normalization_21/batchnorm/ReadVariableOp_21batch_normalization_21/batchnorm/ReadVariableOp_22j
3batch_normalization_21/batchnorm/mul/ReadVariableOp3batch_normalization_21/batchnorm/mul/ReadVariableOp2b
/batch_normalization_22/batchnorm/ReadVariableOp/batch_normalization_22/batchnorm/ReadVariableOp2f
1batch_normalization_22/batchnorm/ReadVariableOp_11batch_normalization_22/batchnorm/ReadVariableOp_12f
1batch_normalization_22/batchnorm/ReadVariableOp_21batch_normalization_22/batchnorm/ReadVariableOp_22j
3batch_normalization_22/batchnorm/mul/ReadVariableOp3batch_normalization_22/batchnorm/mul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2x
:dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2x
:dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp:V R
/
_output_shapes
:���������<Z

_user_specified_nameimage
�
�
+__inference_dense_32_layer_call_fn_20226293

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_32/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_32/kernel/Regularizer/L2LossL2Loss9dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_32/kernel/Regularizer/mulMul*dense_32/kernel/Regularizer/mul/x:output:0+dense_32/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_32/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_22_layer_call_fn_20226327

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
G
+__inference_re_lu_18_layer_call_fn_20226240

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:���������Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_dense_32_layer_call_and_return_conditional_losses_20226307

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_32/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_32/kernel/Regularizer/L2LossL2Loss9dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_32/kernel/Regularizer/mulMul*dense_32/kernel/Regularizer/mul/x:output:0+dense_32/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_32/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_dense_31_layer_call_fn_20226113

inputs1
matmul_readvariableop_resource:	�*-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_31/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�**
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
"dense_31/kernel/Regularizer/L2LossL2Loss9dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0+dense_31/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_31/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������*: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:����������*
 
_user_specified_nameinputs
��
�
E__inference_model_9_layer_call_and_return_conditional_losses_20225694	
image:
'dense_31_matmul_readvariableop_resource:	�*6
(dense_31_biasadd_readvariableop_resource:L
>batch_normalization_21_assignmovingavg_readvariableop_resource:N
@batch_normalization_21_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_21_batchnorm_mul_readvariableop_resource:F
8batch_normalization_21_batchnorm_readvariableop_resource::
'dense_32_matmul_readvariableop_resource:	�7
(dense_32_biasadd_readvariableop_resource:	�M
>batch_normalization_22_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_22_assignmovingavg_1_readvariableop_resource:	�K
<batch_normalization_22_batchnorm_mul_readvariableop_resource:	�G
8batch_normalization_22_batchnorm_readvariableop_resource:	�:
'dense_33_matmul_readvariableop_resource:	�6
(dense_33_biasadd_readvariableop_resource:
identity��&batch_normalization_21/AssignMovingAvg�5batch_normalization_21/AssignMovingAvg/ReadVariableOp�(batch_normalization_21/AssignMovingAvg_1�7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_21/batchnorm/ReadVariableOp�3batch_normalization_21/batchnorm/mul/ReadVariableOp�&batch_normalization_22/AssignMovingAvg�5batch_normalization_22/AssignMovingAvg/ReadVariableOp�(batch_normalization_22/AssignMovingAvg_1�7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_22/batchnorm/ReadVariableOp�3batch_normalization_22/batchnorm/mul/ReadVariableOp�dense_31/BiasAdd/ReadVariableOp�dense_31/MatMul/ReadVariableOp�:dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp�1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp�dense_32/BiasAdd/ReadVariableOp�dense_32/MatMul/ReadVariableOp�:dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp�1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp�dense_33/BiasAdd/ReadVariableOp�dense_33/MatMul/ReadVariableOpa
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  r
flatten_10/ReshapeReshapeimageflatten_10/Const:output:0*
T0*(
_output_shapes
:����������*�
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
dense_31/MatMulMatMulflatten_10/Reshape:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
:dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
+dense_31/dense_31/kernel/Regularizer/L2LossL2LossBdense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: o
*dense_31/dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
(dense_31/dense_31/kernel/Regularizer/mulMul3dense_31/dense_31/kernel/Regularizer/mul/x:output:04dense_31/dense_31/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5batch_normalization_21/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_21/moments/meanMeandense_31/BiasAdd:output:0>batch_normalization_21/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
+batch_normalization_21/moments/StopGradientStopGradient,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes

:�
0batch_normalization_21/moments/SquaredDifferenceSquaredDifferencedense_31/BiasAdd:output:04batch_normalization_21/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
9batch_normalization_21/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_21/moments/varianceMean4batch_normalization_21/moments/SquaredDifference:z:0Bbatch_normalization_21/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
&batch_normalization_21/moments/SqueezeSqueeze,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
(batch_normalization_21/moments/Squeeze_1Squeeze0batch_normalization_21/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_21/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_21/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_21_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_21/AssignMovingAvg/subSub=batch_normalization_21/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_21/moments/Squeeze:output:0*
T0*
_output_shapes
:�
*batch_normalization_21/AssignMovingAvg/mulMul.batch_normalization_21/AssignMovingAvg/sub:z:05batch_normalization_21/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
&batch_normalization_21/AssignMovingAvgAssignSubVariableOp>batch_normalization_21_assignmovingavg_readvariableop_resource.batch_normalization_21/AssignMovingAvg/mul:z:06^batch_normalization_21/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_21/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_21_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_21/AssignMovingAvg_1/subSub?batch_normalization_21/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_21/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
,batch_normalization_21/AssignMovingAvg_1/mulMul0batch_normalization_21/AssignMovingAvg_1/sub:z:07batch_normalization_21/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_21/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_21_assignmovingavg_1_readvariableop_resource0batch_normalization_21/AssignMovingAvg_1/mul:z:08^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_21/batchnorm/addAddV21batch_normalization_21/moments/Squeeze_1:output:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:0;batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_21/batchnorm/mul_1Muldense_31/BiasAdd:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
&batch_normalization_21/batchnorm/mul_2Mul/batch_normalization_21/moments/Squeeze:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:�
/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_21/batchnorm/subSub7batch_normalization_21/batchnorm/ReadVariableOp:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������s
re_lu_18/ReluRelu*batch_normalization_21/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������\
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_9/dropout/MulMulre_lu_18/Relu:activations:0 dropout_9/dropout/Const:output:0*
T0*'
_output_shapes
:���������b
dropout_9/dropout/ShapeShapere_lu_18/Relu:activations:0*
T0*
_output_shapes
:�
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0e
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*'
_output_shapes
:����������
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_32/MatMulMatMuldropout_9/dropout/Mul_1:z:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
+dense_32/dense_32/kernel/Regularizer/L2LossL2LossBdense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: o
*dense_32/dense_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
(dense_32/dense_32/kernel/Regularizer/mulMul3dense_32/dense_32/kernel/Regularizer/mul/x:output:04dense_32/dense_32/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5batch_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_22/moments/meanMeandense_32/BiasAdd:output:0>batch_normalization_22/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_22/moments/StopGradientStopGradient,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_22/moments/SquaredDifferenceSquaredDifferencedense_32/BiasAdd:output:04batch_normalization_22/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
9batch_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_22/moments/varianceMean4batch_normalization_22/moments/SquaredDifference:z:0Bbatch_normalization_22/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_22/moments/SqueezeSqueeze,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_22/moments/Squeeze_1Squeeze0batch_normalization_22/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_22/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_22/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_22_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_22/AssignMovingAvg/subSub=batch_normalization_22/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_22/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_22/AssignMovingAvg/mulMul.batch_normalization_22/AssignMovingAvg/sub:z:05batch_normalization_22/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_22/AssignMovingAvgAssignSubVariableOp>batch_normalization_22_assignmovingavg_readvariableop_resource.batch_normalization_22/AssignMovingAvg/mul:z:06^batch_normalization_22/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_22/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_22_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_22/AssignMovingAvg_1/subSub?batch_normalization_22/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_22/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_22/AssignMovingAvg_1/mulMul0batch_normalization_22/AssignMovingAvg_1/sub:z:07batch_normalization_22/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_22/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_22_assignmovingavg_1_readvariableop_resource0batch_normalization_22/AssignMovingAvg_1/mul:z:08^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_22/batchnorm/addAddV21batch_normalization_22/moments/Squeeze_1:output:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:��
3batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:0;batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_22/batchnorm/mul_1Muldense_32/BiasAdd:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_22/batchnorm/mul_2Mul/batch_normalization_22/moments/Squeeze:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_22/batchnorm/subSub7batch_normalization_22/batchnorm/ReadVariableOp:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������t
re_lu_19/ReluRelu*batch_normalization_22/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_33/MatMulMatMulre_lu_19/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
"dense_31/kernel/Regularizer/L2LossL2Loss9dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0+dense_31/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_32/kernel/Regularizer/L2LossL2Loss9dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_32/kernel/Regularizer/mulMul*dense_32/kernel/Regularizer/mul/x:output:0+dense_32/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_33/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^batch_normalization_21/AssignMovingAvg6^batch_normalization_21/AssignMovingAvg/ReadVariableOp)^batch_normalization_21/AssignMovingAvg_18^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_21/batchnorm/ReadVariableOp4^batch_normalization_21/batchnorm/mul/ReadVariableOp'^batch_normalization_22/AssignMovingAvg6^batch_normalization_22/AssignMovingAvg/ReadVariableOp)^batch_normalization_22/AssignMovingAvg_18^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_22/batchnorm/ReadVariableOp4^batch_normalization_22/batchnorm/mul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp;^dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_31/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp;^dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_32/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<Z: : : : : : : : : : : : : : 2P
&batch_normalization_21/AssignMovingAvg&batch_normalization_21/AssignMovingAvg2n
5batch_normalization_21/AssignMovingAvg/ReadVariableOp5batch_normalization_21/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_21/AssignMovingAvg_1(batch_normalization_21/AssignMovingAvg_12r
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_21/batchnorm/ReadVariableOp/batch_normalization_21/batchnorm/ReadVariableOp2j
3batch_normalization_21/batchnorm/mul/ReadVariableOp3batch_normalization_21/batchnorm/mul/ReadVariableOp2P
&batch_normalization_22/AssignMovingAvg&batch_normalization_22/AssignMovingAvg2n
5batch_normalization_22/AssignMovingAvg/ReadVariableOp5batch_normalization_22/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_22/AssignMovingAvg_1(batch_normalization_22/AssignMovingAvg_12r
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_22/batchnorm/ReadVariableOp/batch_normalization_22/batchnorm/ReadVariableOp2j
3batch_normalization_22/batchnorm/mul/ReadVariableOp3batch_normalization_22/batchnorm/mul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2x
:dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2x
:dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp:V R
/
_output_shapes
:���������<Z

_user_specified_nameimage
�
d
H__inference_flatten_10_layer_call_and_return_conditional_losses_20226099

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
�%
�
9__inference_batch_normalization_22_layer_call_fn_20226361

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
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
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
I
-__inference_flatten_10_layer_call_fn_20226093

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
�
�
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_20226381

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
K
,__inference_dropout_9_layer_call_fn_20226262

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
*__inference_model_9_layer_call_fn_20225509	
image:
'dense_31_matmul_readvariableop_resource:	�*6
(dense_31_biasadd_readvariableop_resource:L
>batch_normalization_21_assignmovingavg_readvariableop_resource:N
@batch_normalization_21_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_21_batchnorm_mul_readvariableop_resource:F
8batch_normalization_21_batchnorm_readvariableop_resource::
'dense_32_matmul_readvariableop_resource:	�7
(dense_32_biasadd_readvariableop_resource:	�M
>batch_normalization_22_assignmovingavg_readvariableop_resource:	�O
@batch_normalization_22_assignmovingavg_1_readvariableop_resource:	�K
<batch_normalization_22_batchnorm_mul_readvariableop_resource:	�G
8batch_normalization_22_batchnorm_readvariableop_resource:	�:
'dense_33_matmul_readvariableop_resource:	�6
(dense_33_biasadd_readvariableop_resource:
identity��&batch_normalization_21/AssignMovingAvg�5batch_normalization_21/AssignMovingAvg/ReadVariableOp�(batch_normalization_21/AssignMovingAvg_1�7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_21/batchnorm/ReadVariableOp�3batch_normalization_21/batchnorm/mul/ReadVariableOp�&batch_normalization_22/AssignMovingAvg�5batch_normalization_22/AssignMovingAvg/ReadVariableOp�(batch_normalization_22/AssignMovingAvg_1�7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_22/batchnorm/ReadVariableOp�3batch_normalization_22/batchnorm/mul/ReadVariableOp�dense_31/BiasAdd/ReadVariableOp�dense_31/MatMul/ReadVariableOp�:dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp�1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp�dense_32/BiasAdd/ReadVariableOp�dense_32/MatMul/ReadVariableOp�:dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp�1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp�dense_33/BiasAdd/ReadVariableOp�dense_33/MatMul/ReadVariableOpa
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  r
flatten_10/ReshapeReshapeimageflatten_10/Const:output:0*
T0*(
_output_shapes
:����������*�
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
dense_31/MatMulMatMulflatten_10/Reshape:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
:dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
+dense_31/dense_31/kernel/Regularizer/L2LossL2LossBdense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: o
*dense_31/dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
(dense_31/dense_31/kernel/Regularizer/mulMul3dense_31/dense_31/kernel/Regularizer/mul/x:output:04dense_31/dense_31/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5batch_normalization_21/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_21/moments/meanMeandense_31/BiasAdd:output:0>batch_normalization_21/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
+batch_normalization_21/moments/StopGradientStopGradient,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes

:�
0batch_normalization_21/moments/SquaredDifferenceSquaredDifferencedense_31/BiasAdd:output:04batch_normalization_21/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
9batch_normalization_21/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_21/moments/varianceMean4batch_normalization_21/moments/SquaredDifference:z:0Bbatch_normalization_21/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
&batch_normalization_21/moments/SqueezeSqueeze,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
(batch_normalization_21/moments/Squeeze_1Squeeze0batch_normalization_21/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_21/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_21/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_21_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
*batch_normalization_21/AssignMovingAvg/subSub=batch_normalization_21/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_21/moments/Squeeze:output:0*
T0*
_output_shapes
:�
*batch_normalization_21/AssignMovingAvg/mulMul.batch_normalization_21/AssignMovingAvg/sub:z:05batch_normalization_21/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
&batch_normalization_21/AssignMovingAvgAssignSubVariableOp>batch_normalization_21_assignmovingavg_readvariableop_resource.batch_normalization_21/AssignMovingAvg/mul:z:06^batch_normalization_21/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_21/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_21_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
,batch_normalization_21/AssignMovingAvg_1/subSub?batch_normalization_21/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_21/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
,batch_normalization_21/AssignMovingAvg_1/mulMul0batch_normalization_21/AssignMovingAvg_1/sub:z:07batch_normalization_21/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
(batch_normalization_21/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_21_assignmovingavg_1_readvariableop_resource0batch_normalization_21/AssignMovingAvg_1/mul:z:08^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_21/batchnorm/addAddV21batch_normalization_21/moments/Squeeze_1:output:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:�
3batch_normalization_21/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_21_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:0;batch_normalization_21/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
&batch_normalization_21/batchnorm/mul_1Muldense_31/BiasAdd:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
&batch_normalization_21/batchnorm/mul_2Mul/batch_normalization_21/moments/Squeeze:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:�
/batch_normalization_21/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_21_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization_21/batchnorm/subSub7batch_normalization_21/batchnorm/ReadVariableOp:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������s
re_lu_18/ReluRelu*batch_normalization_21/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������\
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_9/dropout/MulMulre_lu_18/Relu:activations:0 dropout_9/dropout/Const:output:0*
T0*'
_output_shapes
:���������b
dropout_9/dropout/ShapeShapere_lu_18/Relu:activations:0*
T0*
_output_shapes
:�
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0e
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:����������
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:����������
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*'
_output_shapes
:����������
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_32/MatMulMatMuldropout_9/dropout/Mul_1:z:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
+dense_32/dense_32/kernel/Regularizer/L2LossL2LossBdense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: o
*dense_32/dense_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
(dense_32/dense_32/kernel/Regularizer/mulMul3dense_32/dense_32/kernel/Regularizer/mul/x:output:04dense_32/dense_32/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: 
5batch_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
#batch_normalization_22/moments/meanMeandense_32/BiasAdd:output:0>batch_normalization_22/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+batch_normalization_22/moments/StopGradientStopGradient,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes
:	��
0batch_normalization_22/moments/SquaredDifferenceSquaredDifferencedense_32/BiasAdd:output:04batch_normalization_22/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
9batch_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
'batch_normalization_22/moments/varianceMean4batch_normalization_22/moments/SquaredDifference:z:0Bbatch_normalization_22/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
&batch_normalization_22/moments/SqueezeSqueeze,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
(batch_normalization_22/moments/Squeeze_1Squeeze0batch_normalization_22/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 q
,batch_normalization_22/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5batch_normalization_22/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_22_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*batch_normalization_22/AssignMovingAvg/subSub=batch_normalization_22/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_22/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
*batch_normalization_22/AssignMovingAvg/mulMul.batch_normalization_22/AssignMovingAvg/sub:z:05batch_normalization_22/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
&batch_normalization_22/AssignMovingAvgAssignSubVariableOp>batch_normalization_22_assignmovingavg_readvariableop_resource.batch_normalization_22/AssignMovingAvg/mul:z:06^batch_normalization_22/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_22/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_22_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,batch_normalization_22/AssignMovingAvg_1/subSub?batch_normalization_22/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_22/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
,batch_normalization_22/AssignMovingAvg_1/mulMul0batch_normalization_22/AssignMovingAvg_1/sub:z:07batch_normalization_22/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
(batch_normalization_22/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_22_assignmovingavg_1_readvariableop_resource0batch_normalization_22/AssignMovingAvg_1/mul:z:08^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
$batch_normalization_22/batchnorm/addAddV21batch_normalization_22/moments/Squeeze_1:output:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:��
3batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:0;batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
&batch_normalization_22/batchnorm/mul_1Muldense_32/BiasAdd:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
&batch_normalization_22/batchnorm/mul_2Mul/batch_normalization_22/moments/Squeeze:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$batch_normalization_22/batchnorm/subSub7batch_normalization_22/batchnorm/ReadVariableOp:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������t
re_lu_19/ReluRelu*batch_normalization_22/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_33/MatMulMatMulre_lu_19/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
"dense_31/kernel/Regularizer/L2LossL2Loss9dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0+dense_31/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_32/kernel/Regularizer/L2LossL2Loss9dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_32/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_32/kernel/Regularizer/mulMul*dense_32/kernel/Regularizer/mul/x:output:0+dense_32/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_33/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^batch_normalization_21/AssignMovingAvg6^batch_normalization_21/AssignMovingAvg/ReadVariableOp)^batch_normalization_21/AssignMovingAvg_18^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_21/batchnorm/ReadVariableOp4^batch_normalization_21/batchnorm/mul/ReadVariableOp'^batch_normalization_22/AssignMovingAvg6^batch_normalization_22/AssignMovingAvg/ReadVariableOp)^batch_normalization_22/AssignMovingAvg_18^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_22/batchnorm/ReadVariableOp4^batch_normalization_22/batchnorm/mul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp;^dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_31/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp;^dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_32/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������<Z: : : : : : : : : : : : : : 2P
&batch_normalization_21/AssignMovingAvg&batch_normalization_21/AssignMovingAvg2n
5batch_normalization_21/AssignMovingAvg/ReadVariableOp5batch_normalization_21/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_21/AssignMovingAvg_1(batch_normalization_21/AssignMovingAvg_12r
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_21/batchnorm/ReadVariableOp/batch_normalization_21/batchnorm/ReadVariableOp2j
3batch_normalization_21/batchnorm/mul/ReadVariableOp3batch_normalization_21/batchnorm/mul/ReadVariableOp2P
&batch_normalization_22/AssignMovingAvg&batch_normalization_22/AssignMovingAvg2n
5batch_normalization_22/AssignMovingAvg/ReadVariableOp5batch_normalization_22/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_22/AssignMovingAvg_1(batch_normalization_22/AssignMovingAvg_12r
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_22/batchnorm/ReadVariableOp/batch_normalization_22/batchnorm/ReadVariableOp2j
3batch_normalization_22/batchnorm/mul/ReadVariableOp3batch_normalization_22/batchnorm/mul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2x
:dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:dense_31/dense_31/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2x
:dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp:dense_32/dense_32/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp1dense_32/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp:V R
/
_output_shapes
:���������<Z

_user_specified_nameimage
�
e
G__inference_dropout_9_layer_call_and_return_conditional_losses_20226267

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_re_lu_19_layer_call_and_return_conditional_losses_20226425

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:����������[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_0_20226454M
:dense_31_kernel_regularizer_l2loss_readvariableop_resource:	�*
identity��1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp�
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_31_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	�**
dtype0�
"dense_31/kernel/Regularizer/L2LossL2Loss9dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0+dense_31/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_31/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_31/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp
�
�
9__inference_batch_normalization_21_layer_call_fn_20226147

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
J
,__inference_dropout_9_layer_call_fn_20226250

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
F__inference_dense_33_layer_call_and_return_conditional_losses_20226445

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
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
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
F__inference_re_lu_18_layer_call_and_return_conditional_losses_20226245

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:���������Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_20226201

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_dense_31_layer_call_and_return_conditional_losses_20226127

inputs1
matmul_readvariableop_resource:	�*-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_31/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�**
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�**
dtype0�
"dense_31/kernel/Regularizer/L2LossL2Loss9dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0+dense_31/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_31/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������*: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp1dense_31/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:����������*
 
_user_specified_nameinputs
�
G
+__inference_re_lu_19_layer_call_fn_20226420

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:����������[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�	-
saver_filename:0
Identity:0Identity_418"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
image6
serving_default_image:0���������<Z<
dense_330
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
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
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

 kernel
!bias"
_tf_keras_layer
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(axis
	)gamma
*beta
+moving_mean
,moving_variance"
_tf_keras_layer
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses
9_random_generator"
_tf_keras_layer
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Haxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance"
_tf_keras_layer
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias"
_tf_keras_layer
�
 0
!1
)2
*3
+4
,5
@6
A7
I8
J9
K10
L11
Y12
Z13"
trackable_list_wrapper
f
 0
!1
)2
*3
@4
A5
I6
J7
Y8
Z9"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
btrace_0
ctrace_1
dtrace_2
etrace_32�
*__inference_model_9_layer_call_fn_20224872
*__inference_model_9_layer_call_fn_20225816
*__inference_model_9_layer_call_fn_20225918
*__inference_model_9_layer_call_fn_20225509�
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
 zbtrace_0zctrace_1zdtrace_2zetrace_3
�
ftrace_0
gtrace_1
htrace_2
itrace_32�
E__inference_model_9_layer_call_and_return_conditional_losses_20225985
E__inference_model_9_layer_call_and_return_conditional_losses_20226087
E__inference_model_9_layer_call_and_return_conditional_losses_20225584
E__inference_model_9_layer_call_and_return_conditional_losses_20225694�
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
 zftrace_0zgtrace_1zhtrace_2zitrace_3
�B�
#__inference__wrapped_model_20224576image"�
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
jiter

kbeta_1

lbeta_2
	mdecay m�!m�)m�*m�@m�Am�Im�Jm�Ym�Zm� v�!v�)v�*v�@v�Av�Iv�Jv�Yv�Zv�"
	optimizer
,
nserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
ttrace_02�
-__inference_flatten_10_layer_call_fn_20226093�
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
 zttrace_0
�
utrace_02�
H__inference_flatten_10_layer_call_and_return_conditional_losses_20226099�
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
 zutrace_0
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
'
[0"
trackable_list_wrapper
�
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
{trace_02�
+__inference_dense_31_layer_call_fn_20226113�
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
 z{trace_0
�
|trace_02�
F__inference_dense_31_layer_call_and_return_conditional_losses_20226127�
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
 z|trace_0
": 	�*2dense_31/kernel
:2dense_31/bias
<
)0
*1
+2
,3"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_21_layer_call_fn_20226147
9__inference_batch_normalization_21_layer_call_fn_20226181�
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
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_20226201
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_20226235�
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
*:(2batch_normalization_21/gamma
):'2batch_normalization_21/beta
2:0 (2"batch_normalization_21/moving_mean
6:4 (2&batch_normalization_21/moving_variance
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
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_re_lu_18_layer_call_fn_20226240�
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
F__inference_re_lu_18_layer_call_and_return_conditional_losses_20226245�
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
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_9_layer_call_fn_20226250
,__inference_dropout_9_layer_call_fn_20226262�
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
G__inference_dropout_9_layer_call_and_return_conditional_losses_20226267
G__inference_dropout_9_layer_call_and_return_conditional_losses_20226279�
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
_generic_user_object
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
'
\0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_32_layer_call_fn_20226293�
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
F__inference_dense_32_layer_call_and_return_conditional_losses_20226307�
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
": 	�2dense_32/kernel
:�2dense_32/bias
<
I0
J1
K2
L3"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_22_layer_call_fn_20226327
9__inference_batch_normalization_22_layer_call_fn_20226361�
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
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_20226381
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_20226415�
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
+:)�2batch_normalization_22/gamma
*:(�2batch_normalization_22/beta
3:1� (2"batch_normalization_22/moving_mean
7:5� (2&batch_normalization_22/moving_variance
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
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_re_lu_19_layer_call_fn_20226420�
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
F__inference_re_lu_19_layer_call_and_return_conditional_losses_20226425�
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
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_33_layer_call_fn_20226435�
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
F__inference_dense_33_layer_call_and_return_conditional_losses_20226445�
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
": 	�2dense_33/kernel
:2dense_33/bias
�
�trace_02�
__inference_loss_fn_0_20226454�
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
__inference_loss_fn_1_20226463�
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
<
+0
,1
K2
L3"
trackable_list_wrapper
f
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
9"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_model_9_layer_call_fn_20224872image"�
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
*__inference_model_9_layer_call_fn_20225816inputs"�
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
*__inference_model_9_layer_call_fn_20225918inputs"�
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
*__inference_model_9_layer_call_fn_20225509image"�
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
E__inference_model_9_layer_call_and_return_conditional_losses_20225985inputs"�
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
E__inference_model_9_layer_call_and_return_conditional_losses_20226087inputs"�
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
E__inference_model_9_layer_call_and_return_conditional_losses_20225584image"�
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
E__inference_model_9_layer_call_and_return_conditional_losses_20225694image"�
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
&__inference_signature_wrapper_20225741image"�
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
-__inference_flatten_10_layer_call_fn_20226093inputs"�
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
H__inference_flatten_10_layer_call_and_return_conditional_losses_20226099inputs"�
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
[0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_31_layer_call_fn_20226113inputs"�
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
F__inference_dense_31_layer_call_and_return_conditional_losses_20226127inputs"�
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
+0
,1"
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
9__inference_batch_normalization_21_layer_call_fn_20226147inputs"�
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
9__inference_batch_normalization_21_layer_call_fn_20226181inputs"�
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
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_20226201inputs"�
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
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_20226235inputs"�
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
+__inference_re_lu_18_layer_call_fn_20226240inputs"�
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
F__inference_re_lu_18_layer_call_and_return_conditional_losses_20226245inputs"�
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
,__inference_dropout_9_layer_call_fn_20226250inputs"�
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
,__inference_dropout_9_layer_call_fn_20226262inputs"�
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
G__inference_dropout_9_layer_call_and_return_conditional_losses_20226267inputs"�
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
G__inference_dropout_9_layer_call_and_return_conditional_losses_20226279inputs"�
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
\0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_32_layer_call_fn_20226293inputs"�
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
F__inference_dense_32_layer_call_and_return_conditional_losses_20226307inputs"�
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
K0
L1"
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
9__inference_batch_normalization_22_layer_call_fn_20226327inputs"�
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
9__inference_batch_normalization_22_layer_call_fn_20226361inputs"�
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
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_20226381inputs"�
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
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_20226415inputs"�
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
+__inference_re_lu_19_layer_call_fn_20226420inputs"�
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
F__inference_re_lu_19_layer_call_and_return_conditional_losses_20226425inputs"�
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
+__inference_dense_33_layer_call_fn_20226435inputs"�
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
F__inference_dense_33_layer_call_and_return_conditional_losses_20226445inputs"�
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
__inference_loss_fn_0_20226454"�
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
__inference_loss_fn_1_20226463"�
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
':%	�*2Adam/dense_31/kernel/m
 :2Adam/dense_31/bias/m
/:-2#Adam/batch_normalization_21/gamma/m
.:,2"Adam/batch_normalization_21/beta/m
':%	�2Adam/dense_32/kernel/m
!:�2Adam/dense_32/bias/m
0:.�2#Adam/batch_normalization_22/gamma/m
/:-�2"Adam/batch_normalization_22/beta/m
':%	�2Adam/dense_33/kernel/m
 :2Adam/dense_33/bias/m
':%	�*2Adam/dense_31/kernel/v
 :2Adam/dense_31/bias/v
/:-2#Adam/batch_normalization_21/gamma/v
.:,2"Adam/batch_normalization_21/beta/v
':%	�2Adam/dense_32/kernel/v
!:�2Adam/dense_32/bias/v
0:.�2#Adam/batch_normalization_22/gamma/v
/:-�2"Adam/batch_normalization_22/beta/v
':%	�2Adam/dense_33/kernel/v
 :2Adam/dense_33/bias/v�
#__inference__wrapped_model_20224576} !,)+*@ALIKJYZ6�3
,�)
'�$
image���������<Z
� "3�0
.
dense_33"�
dense_33����������
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_20226201b,)+*3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_20226235b+,)*3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
9__inference_batch_normalization_21_layer_call_fn_20226147U,)+*3�0
)�&
 �
inputs���������
p 
� "�����������
9__inference_batch_normalization_21_layer_call_fn_20226181U+,)*3�0
)�&
 �
inputs���������
p
� "�����������
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_20226381dLIKJ4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_20226415dKLIJ4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
9__inference_batch_normalization_22_layer_call_fn_20226327WLIKJ4�1
*�'
!�
inputs����������
p 
� "������������
9__inference_batch_normalization_22_layer_call_fn_20226361WKLIJ4�1
*�'
!�
inputs����������
p
� "������������
F__inference_dense_31_layer_call_and_return_conditional_losses_20226127] !0�-
&�#
!�
inputs����������*
� "%�"
�
0���������
� 
+__inference_dense_31_layer_call_fn_20226113P !0�-
&�#
!�
inputs����������*
� "�����������
F__inference_dense_32_layer_call_and_return_conditional_losses_20226307]@A/�,
%�"
 �
inputs���������
� "&�#
�
0����������
� 
+__inference_dense_32_layer_call_fn_20226293P@A/�,
%�"
 �
inputs���������
� "������������
F__inference_dense_33_layer_call_and_return_conditional_losses_20226445]YZ0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� 
+__inference_dense_33_layer_call_fn_20226435PYZ0�-
&�#
!�
inputs����������
� "�����������
G__inference_dropout_9_layer_call_and_return_conditional_losses_20226267\3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
G__inference_dropout_9_layer_call_and_return_conditional_losses_20226279\3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� 
,__inference_dropout_9_layer_call_fn_20226250O3�0
)�&
 �
inputs���������
p 
� "����������
,__inference_dropout_9_layer_call_fn_20226262O3�0
)�&
 �
inputs���������
p
� "�����������
H__inference_flatten_10_layer_call_and_return_conditional_losses_20226099a7�4
-�*
(�%
inputs���������<Z
� "&�#
�
0����������*
� �
-__inference_flatten_10_layer_call_fn_20226093T7�4
-�*
(�%
inputs���������<Z
� "�����������*=
__inference_loss_fn_0_20226454 �

� 
� "� =
__inference_loss_fn_1_20226463@�

� 
� "� �
E__inference_model_9_layer_call_and_return_conditional_losses_20225584w !,)+*@ALIKJYZ>�;
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
E__inference_model_9_layer_call_and_return_conditional_losses_20225694w !+,)*@AKLIJYZ>�;
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
E__inference_model_9_layer_call_and_return_conditional_losses_20225985x !,)+*@ALIKJYZ?�<
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
E__inference_model_9_layer_call_and_return_conditional_losses_20226087x !+,)*@AKLIJYZ?�<
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
*__inference_model_9_layer_call_fn_20224872j !,)+*@ALIKJYZ>�;
4�1
'�$
image���������<Z
p 

 
� "�����������
*__inference_model_9_layer_call_fn_20225509j !+,)*@AKLIJYZ>�;
4�1
'�$
image���������<Z
p

 
� "�����������
*__inference_model_9_layer_call_fn_20225816k !,)+*@ALIKJYZ?�<
5�2
(�%
inputs���������<Z
p 

 
� "�����������
*__inference_model_9_layer_call_fn_20225918k !+,)*@AKLIJYZ?�<
5�2
(�%
inputs���������<Z
p

 
� "�����������
F__inference_re_lu_18_layer_call_and_return_conditional_losses_20226245X/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� z
+__inference_re_lu_18_layer_call_fn_20226240K/�,
%�"
 �
inputs���������
� "�����������
F__inference_re_lu_19_layer_call_and_return_conditional_losses_20226425Z0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
+__inference_re_lu_19_layer_call_fn_20226420M0�-
&�#
!�
inputs����������
� "������������
&__inference_signature_wrapper_20225741� !,)+*@ALIKJYZ?�<
� 
5�2
0
image'�$
image���������<Z"3�0
.
dense_33"�
dense_33���������