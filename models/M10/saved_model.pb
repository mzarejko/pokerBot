ê

§
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
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
list(type)(0
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
list(type)(0
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
¾
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
executor_typestring 
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
;
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12unknown8ô
|
dense_92/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_92/kernel
u
#dense_92/kernel/Read/ReadVariableOpReadVariableOpdense_92/kernel* 
_output_shapes
:
*
dtype0
s
dense_92/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_92/bias
l
!dense_92/bias/Read/ReadVariableOpReadVariableOpdense_92/bias*
_output_shapes	
:*
dtype0
|
dense_93/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_93/kernel
u
#dense_93/kernel/Read/ReadVariableOpReadVariableOpdense_93/kernel* 
_output_shapes
:
*
dtype0
s
dense_93/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_93/bias
l
!dense_93/bias/Read/ReadVariableOpReadVariableOpdense_93/bias*
_output_shapes	
:*
dtype0
|
dense_94/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_94/kernel
u
#dense_94/kernel/Read/ReadVariableOpReadVariableOpdense_94/kernel* 
_output_shapes
:
*
dtype0
s
dense_94/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_94/bias
l
!dense_94/bias/Read/ReadVariableOpReadVariableOpdense_94/bias*
_output_shapes	
:*
dtype0

batch_normalization_23/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_23/gamma

0batch_normalization_23/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_23/gamma*
_output_shapes	
:*
dtype0

batch_normalization_23/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_23/beta

/batch_normalization_23/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_23/beta*
_output_shapes	
:*
dtype0

"batch_normalization_23/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_23/moving_mean

6batch_normalization_23/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_23/moving_mean*
_output_shapes	
:*
dtype0
¥
&batch_normalization_23/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_23/moving_variance

:batch_normalization_23/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_23/moving_variance*
_output_shapes	
:*
dtype0
{
dense_95/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_95/kernel
t
#dense_95/kernel/Read/ReadVariableOpReadVariableOpdense_95/kernel*
_output_shapes
:	*
dtype0
r
dense_95/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_95/bias
k
!dense_95/bias/Read/ReadVariableOpReadVariableOpdense_95/bias*
_output_shapes
:*
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
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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

Adam/dense_92/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_92/kernel/m

*Adam/dense_92/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_92/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_92/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_92/bias/m
z
(Adam/dense_92/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_92/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_93/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_93/kernel/m

*Adam/dense_93/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_93/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_93/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_93/bias/m
z
(Adam/dense_93/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_93/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_94/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_94/kernel/m

*Adam/dense_94/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_94/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_94/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_94/bias/m
z
(Adam/dense_94/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_94/bias/m*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_23/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_23/gamma/m

7Adam/batch_normalization_23/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_23/gamma/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_23/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_23/beta/m

6Adam/batch_normalization_23/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_23/beta/m*
_output_shapes	
:*
dtype0

Adam/dense_95/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_95/kernel/m

*Adam/dense_95/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_95/kernel/m*
_output_shapes
:	*
dtype0

Adam/dense_95/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_95/bias/m
y
(Adam/dense_95/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_95/bias/m*
_output_shapes
:*
dtype0

Adam/dense_92/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_92/kernel/v

*Adam/dense_92/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_92/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_92/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_92/bias/v
z
(Adam/dense_92/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_92/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_93/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_93/kernel/v

*Adam/dense_93/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_93/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_93/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_93/bias/v
z
(Adam/dense_93/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_93/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_94/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_94/kernel/v

*Adam/dense_94/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_94/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_94/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_94/bias/v
z
(Adam/dense_94/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_94/bias/v*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_23/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_23/gamma/v

7Adam/batch_normalization_23/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_23/gamma/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_23/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_23/beta/v

6Adam/batch_normalization_23/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_23/beta/v*
_output_shapes	
:*
dtype0

Adam/dense_95/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_95/kernel/v

*Adam/dense_95/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_95/kernel/v*
_output_shapes
:	*
dtype0

Adam/dense_95/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_95/bias/v
y
(Adam/dense_95/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_95/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
¼:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*÷9
valueí9Bê9 Bã9
Î
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
 
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api

$axis
	%gamma
&beta
'moving_mean
(moving_variance
)regularization_losses
*	variables
+trainable_variables
,	keras_api
h

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
ô
3iter

4beta_1

5beta_2
	6decay
7learning_ratem`mambmcmdme%mf&mg-mh.mivjvkvlvmvnvo%vp&vq-vr.vs
 
V
0
1
2
3
4
5
%6
&7
'8
(9
-10
.11
F
0
1
2
3
4
5
%6
&7
-8
.9
­

8layers
9metrics
	regularization_losses
:non_trainable_variables

	variables
;layer_regularization_losses
<layer_metrics
trainable_variables
 
 
 
 
­
=metrics

>layers
regularization_losses
?non_trainable_variables
	variables
@layer_regularization_losses
Alayer_metrics
trainable_variables
[Y
VARIABLE_VALUEdense_92/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_92/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
Bmetrics

Clayers
regularization_losses
Dnon_trainable_variables
	variables
Elayer_regularization_losses
Flayer_metrics
trainable_variables
[Y
VARIABLE_VALUEdense_93/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_93/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
Gmetrics

Hlayers
regularization_losses
Inon_trainable_variables
	variables
Jlayer_regularization_losses
Klayer_metrics
trainable_variables
[Y
VARIABLE_VALUEdense_94/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_94/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
Lmetrics

Mlayers
 regularization_losses
Nnon_trainable_variables
!	variables
Olayer_regularization_losses
Player_metrics
"trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_23/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_23/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_23/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_23/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1
'2
(3

%0
&1
­
Qmetrics

Rlayers
)regularization_losses
Snon_trainable_variables
*	variables
Tlayer_regularization_losses
Ulayer_metrics
+trainable_variables
[Y
VARIABLE_VALUEdense_95/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_95/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

-0
.1
­
Vmetrics

Wlayers
/regularization_losses
Xnon_trainable_variables
0	variables
Ylayer_regularization_losses
Zlayer_metrics
1trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
1
0
1
2
3
4
5
6

[0

'0
(1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

'0
(1
 
 
 
 
 
 
 
4
	\total
	]count
^	variables
_	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

\0
]1

^	variables
~|
VARIABLE_VALUEAdam/dense_92/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_92/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_93/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_93/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_94/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_94/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_23/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_23/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_95/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_95/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_92/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_92/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_93/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_93/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_94/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_94/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_23/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_23/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_95/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_95/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_24Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ4
Õ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_24dense_92/kerneldense_92/biasdense_93/kerneldense_93/biasdense_94/kerneldense_94/bias&batch_normalization_23/moving_variancebatch_normalization_23/gamma"batch_normalization_23/moving_meanbatch_normalization_23/betadense_95/kerneldense_95/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_signature_wrapper_226664501
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
­
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_92/kernel/Read/ReadVariableOp!dense_92/bias/Read/ReadVariableOp#dense_93/kernel/Read/ReadVariableOp!dense_93/bias/Read/ReadVariableOp#dense_94/kernel/Read/ReadVariableOp!dense_94/bias/Read/ReadVariableOp0batch_normalization_23/gamma/Read/ReadVariableOp/batch_normalization_23/beta/Read/ReadVariableOp6batch_normalization_23/moving_mean/Read/ReadVariableOp:batch_normalization_23/moving_variance/Read/ReadVariableOp#dense_95/kernel/Read/ReadVariableOp!dense_95/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_92/kernel/m/Read/ReadVariableOp(Adam/dense_92/bias/m/Read/ReadVariableOp*Adam/dense_93/kernel/m/Read/ReadVariableOp(Adam/dense_93/bias/m/Read/ReadVariableOp*Adam/dense_94/kernel/m/Read/ReadVariableOp(Adam/dense_94/bias/m/Read/ReadVariableOp7Adam/batch_normalization_23/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_23/beta/m/Read/ReadVariableOp*Adam/dense_95/kernel/m/Read/ReadVariableOp(Adam/dense_95/bias/m/Read/ReadVariableOp*Adam/dense_92/kernel/v/Read/ReadVariableOp(Adam/dense_92/bias/v/Read/ReadVariableOp*Adam/dense_93/kernel/v/Read/ReadVariableOp(Adam/dense_93/bias/v/Read/ReadVariableOp*Adam/dense_94/kernel/v/Read/ReadVariableOp(Adam/dense_94/bias/v/Read/ReadVariableOp7Adam/batch_normalization_23/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_23/beta/v/Read/ReadVariableOp*Adam/dense_95/kernel/v/Read/ReadVariableOp(Adam/dense_95/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_save_226664988
	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_92/kerneldense_92/biasdense_93/kerneldense_93/biasdense_94/kerneldense_94/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_variancedense_95/kerneldense_95/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_92/kernel/mAdam/dense_92/bias/mAdam/dense_93/kernel/mAdam/dense_93/bias/mAdam/dense_94/kernel/mAdam/dense_94/bias/m#Adam/batch_normalization_23/gamma/m"Adam/batch_normalization_23/beta/mAdam/dense_95/kernel/mAdam/dense_95/bias/mAdam/dense_92/kernel/vAdam/dense_92/bias/vAdam/dense_93/kernel/vAdam/dense_93/bias/vAdam/dense_94/kernel/vAdam/dense_94/bias/v#Adam/batch_normalization_23/gamma/v"Adam/batch_normalization_23/beta/vAdam/dense_95/kernel/vAdam/dense_95/bias/v*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference__traced_restore_226665115Ì
Á
­
:__inference_batch_normalization_23_layer_call_fn_226664815

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_batch_normalization_23_layer_call_and_return_conditional_losses_2266640952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
J
.__inference_flatten_23_layer_call_fn_226664686

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_flatten_23_layer_call_and_return_conditional_losses_2266641492
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ4:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ªF
®	
G__inference_model_23_layer_call_and_return_conditional_losses_226664617

inputs+
'dense_92_matmul_readvariableop_resource,
(dense_92_biasadd_readvariableop_resource+
'dense_93_matmul_readvariableop_resource,
(dense_93_biasadd_readvariableop_resource+
'dense_94_matmul_readvariableop_resource,
(dense_94_biasadd_readvariableop_resource<
8batch_normalization_23_batchnorm_readvariableop_resource@
<batch_normalization_23_batchnorm_mul_readvariableop_resource>
:batch_normalization_23_batchnorm_readvariableop_1_resource>
:batch_normalization_23_batchnorm_readvariableop_2_resource+
'dense_95_matmul_readvariableop_resource,
(dense_95_biasadd_readvariableop_resource
identity¢/batch_normalization_23/batchnorm/ReadVariableOp¢1batch_normalization_23/batchnorm/ReadVariableOp_1¢1batch_normalization_23/batchnorm/ReadVariableOp_2¢3batch_normalization_23/batchnorm/mul/ReadVariableOp¢dense_92/BiasAdd/ReadVariableOp¢dense_92/MatMul/ReadVariableOp¢dense_93/BiasAdd/ReadVariableOp¢dense_93/MatMul/ReadVariableOp¢dense_94/BiasAdd/ReadVariableOp¢dense_94/MatMul/ReadVariableOp¢dense_95/BiasAdd/ReadVariableOp¢dense_95/MatMul/ReadVariableOpu
flatten_23/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_23/Const
flatten_23/ReshapeReshapeinputsflatten_23/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_23/Reshapeª
dense_92/MatMul/ReadVariableOpReadVariableOp'dense_92_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_92/MatMul/ReadVariableOp¤
dense_92/MatMulMatMulflatten_23/Reshape:output:0&dense_92/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_92/MatMul¨
dense_92/BiasAdd/ReadVariableOpReadVariableOp(dense_92_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_92/BiasAdd/ReadVariableOp¦
dense_92/BiasAddBiasAdddense_92/MatMul:product:0'dense_92/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_92/BiasAddt
dense_92/ReluReludense_92/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_92/Reluª
dense_93/MatMul/ReadVariableOpReadVariableOp'dense_93_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_93/MatMul/ReadVariableOp¤
dense_93/MatMulMatMuldense_92/Relu:activations:0&dense_93/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_93/MatMul¨
dense_93/BiasAdd/ReadVariableOpReadVariableOp(dense_93_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_93/BiasAdd/ReadVariableOp¦
dense_93/BiasAddBiasAdddense_93/MatMul:product:0'dense_93/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_93/BiasAddt
dense_93/ReluReludense_93/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_93/Reluª
dense_94/MatMul/ReadVariableOpReadVariableOp'dense_94_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_94/MatMul/ReadVariableOp¤
dense_94/MatMulMatMuldense_93/Relu:activations:0&dense_94/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_94/MatMul¨
dense_94/BiasAdd/ReadVariableOpReadVariableOp(dense_94_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_94/BiasAdd/ReadVariableOp¦
dense_94/BiasAddBiasAdddense_94/MatMul:product:0'dense_94/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_94/BiasAddt
dense_94/ReluReludense_94/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_94/ReluØ
/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype021
/batch_normalization_23/batchnorm/ReadVariableOp
&batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_23/batchnorm/add/yå
$batch_normalization_23/batchnorm/addAddV27batch_normalization_23/batchnorm/ReadVariableOp:value:0/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2&
$batch_normalization_23/batchnorm/add©
&batch_normalization_23/batchnorm/RsqrtRsqrt(batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_23/batchnorm/Rsqrtä
3batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype025
3batch_normalization_23/batchnorm/mul/ReadVariableOpâ
$batch_normalization_23/batchnorm/mulMul*batch_normalization_23/batchnorm/Rsqrt:y:0;batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2&
$batch_normalization_23/batchnorm/mulÑ
&batch_normalization_23/batchnorm/mul_1Muldense_94/Relu:activations:0(batch_normalization_23/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_23/batchnorm/mul_1Þ
1batch_normalization_23/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_23_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype023
1batch_normalization_23/batchnorm/ReadVariableOp_1â
&batch_normalization_23/batchnorm/mul_2Mul9batch_normalization_23/batchnorm/ReadVariableOp_1:value:0(batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_23/batchnorm/mul_2Þ
1batch_normalization_23/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_23_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype023
1batch_normalization_23/batchnorm/ReadVariableOp_2à
$batch_normalization_23/batchnorm/subSub9batch_normalization_23/batchnorm/ReadVariableOp_2:value:0*batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2&
$batch_normalization_23/batchnorm/subâ
&batch_normalization_23/batchnorm/add_1AddV2*batch_normalization_23/batchnorm/mul_1:z:0(batch_normalization_23/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_23/batchnorm/add_1©
dense_95/MatMul/ReadVariableOpReadVariableOp'dense_95_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_95/MatMul/ReadVariableOp²
dense_95/MatMulMatMul*batch_normalization_23/batchnorm/add_1:z:0&dense_95/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_95/MatMul§
dense_95/BiasAdd/ReadVariableOpReadVariableOp(dense_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_95/BiasAdd/ReadVariableOp¥
dense_95/BiasAddBiasAdddense_95/MatMul:product:0'dense_95/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_95/BiasAdd|
dense_95/SoftmaxSoftmaxdense_95/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_95/SoftmaxÊ
IdentityIdentitydense_95/Softmax:softmax:00^batch_normalization_23/batchnorm/ReadVariableOp2^batch_normalization_23/batchnorm/ReadVariableOp_12^batch_normalization_23/batchnorm/ReadVariableOp_24^batch_normalization_23/batchnorm/mul/ReadVariableOp ^dense_92/BiasAdd/ReadVariableOp^dense_92/MatMul/ReadVariableOp ^dense_93/BiasAdd/ReadVariableOp^dense_93/MatMul/ReadVariableOp ^dense_94/BiasAdd/ReadVariableOp^dense_94/MatMul/ReadVariableOp ^dense_95/BiasAdd/ReadVariableOp^dense_95/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2b
/batch_normalization_23/batchnorm/ReadVariableOp/batch_normalization_23/batchnorm/ReadVariableOp2f
1batch_normalization_23/batchnorm/ReadVariableOp_11batch_normalization_23/batchnorm/ReadVariableOp_12f
1batch_normalization_23/batchnorm/ReadVariableOp_21batch_normalization_23/batchnorm/ReadVariableOp_22j
3batch_normalization_23/batchnorm/mul/ReadVariableOp3batch_normalization_23/batchnorm/mul/ReadVariableOp2B
dense_92/BiasAdd/ReadVariableOpdense_92/BiasAdd/ReadVariableOp2@
dense_92/MatMul/ReadVariableOpdense_92/MatMul/ReadVariableOp2B
dense_93/BiasAdd/ReadVariableOpdense_93/BiasAdd/ReadVariableOp2@
dense_93/MatMul/ReadVariableOpdense_93/MatMul/ReadVariableOp2B
dense_94/BiasAdd/ReadVariableOpdense_94/BiasAdd/ReadVariableOp2@
dense_94/MatMul/ReadVariableOpdense_94/MatMul/ReadVariableOp2B
dense_95/BiasAdd/ReadVariableOpdense_95/BiasAdd/ReadVariableOp2@
dense_95/MatMul/ReadVariableOpdense_95/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
	

'__inference_signature_wrapper_226664501
input_24
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinput_24unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__wrapped_model_2266639992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
input_24
ü	
à
G__inference_dense_95_layer_call_and_return_conditional_losses_226664284

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í!
û
G__inference_model_23_layer_call_and_return_conditional_losses_226664372

inputs
dense_92_226664342
dense_92_226664344
dense_93_226664347
dense_93_226664349
dense_94_226664352
dense_94_226664354$
 batch_normalization_23_226664357$
 batch_normalization_23_226664359$
 batch_normalization_23_226664361$
 batch_normalization_23_226664363
dense_95_226664366
dense_95_226664368
identity¢.batch_normalization_23/StatefulPartitionedCall¢ dense_92/StatefulPartitionedCall¢ dense_93/StatefulPartitionedCall¢ dense_94/StatefulPartitionedCall¢ dense_95/StatefulPartitionedCallÞ
flatten_23/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_flatten_23_layer_call_and_return_conditional_losses_2266641492
flatten_23/PartitionedCall»
 dense_92/StatefulPartitionedCallStatefulPartitionedCall#flatten_23/PartitionedCall:output:0dense_92_226664342dense_92_226664344*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_92_layer_call_and_return_conditional_losses_2266641682"
 dense_92/StatefulPartitionedCallÁ
 dense_93/StatefulPartitionedCallStatefulPartitionedCall)dense_92/StatefulPartitionedCall:output:0dense_93_226664347dense_93_226664349*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_93_layer_call_and_return_conditional_losses_2266641952"
 dense_93/StatefulPartitionedCallÁ
 dense_94/StatefulPartitionedCallStatefulPartitionedCall)dense_93/StatefulPartitionedCall:output:0dense_94_226664352dense_94_226664354*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_94_layer_call_and_return_conditional_losses_2266642222"
 dense_94/StatefulPartitionedCallÍ
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall)dense_94/StatefulPartitionedCall:output:0 batch_normalization_23_226664357 batch_normalization_23_226664359 batch_normalization_23_226664361 batch_normalization_23_226664363*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_batch_normalization_23_layer_call_and_return_conditional_losses_22666409520
.batch_normalization_23/StatefulPartitionedCallÎ
 dense_95/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0dense_95_226664366dense_95_226664368*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_95_layer_call_and_return_conditional_losses_2266642842"
 dense_95/StatefulPartitionedCallº
IdentityIdentity)dense_95/StatefulPartitionedCall:output:0/^batch_normalization_23/StatefulPartitionedCall!^dense_92/StatefulPartitionedCall!^dense_93/StatefulPartitionedCall!^dense_94/StatefulPartitionedCall!^dense_95/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2D
 dense_92/StatefulPartitionedCall dense_92/StatefulPartitionedCall2D
 dense_93/StatefulPartitionedCall dense_93/StatefulPartitionedCall2D
 dense_94/StatefulPartitionedCall dense_94/StatefulPartitionedCall2D
 dense_95/StatefulPartitionedCall dense_95/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ÜT

"__inference__traced_save_226664988
file_prefix.
*savev2_dense_92_kernel_read_readvariableop,
(savev2_dense_92_bias_read_readvariableop.
*savev2_dense_93_kernel_read_readvariableop,
(savev2_dense_93_bias_read_readvariableop.
*savev2_dense_94_kernel_read_readvariableop,
(savev2_dense_94_bias_read_readvariableop;
7savev2_batch_normalization_23_gamma_read_readvariableop:
6savev2_batch_normalization_23_beta_read_readvariableopA
=savev2_batch_normalization_23_moving_mean_read_readvariableopE
Asavev2_batch_normalization_23_moving_variance_read_readvariableop.
*savev2_dense_95_kernel_read_readvariableop,
(savev2_dense_95_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_92_kernel_m_read_readvariableop3
/savev2_adam_dense_92_bias_m_read_readvariableop5
1savev2_adam_dense_93_kernel_m_read_readvariableop3
/savev2_adam_dense_93_bias_m_read_readvariableop5
1savev2_adam_dense_94_kernel_m_read_readvariableop3
/savev2_adam_dense_94_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_23_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_23_beta_m_read_readvariableop5
1savev2_adam_dense_95_kernel_m_read_readvariableop3
/savev2_adam_dense_95_bias_m_read_readvariableop5
1savev2_adam_dense_92_kernel_v_read_readvariableop3
/savev2_adam_dense_92_bias_v_read_readvariableop5
1savev2_adam_dense_93_kernel_v_read_readvariableop3
/savev2_adam_dense_93_bias_v_read_readvariableop5
1savev2_adam_dense_94_kernel_v_read_readvariableop3
/savev2_adam_dense_94_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_23_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_23_beta_v_read_readvariableop5
1savev2_adam_dense_95_kernel_v_read_readvariableop3
/savev2_adam_dense_95_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*¡
valueB(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesØ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÚ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_92_kernel_read_readvariableop(savev2_dense_92_bias_read_readvariableop*savev2_dense_93_kernel_read_readvariableop(savev2_dense_93_bias_read_readvariableop*savev2_dense_94_kernel_read_readvariableop(savev2_dense_94_bias_read_readvariableop7savev2_batch_normalization_23_gamma_read_readvariableop6savev2_batch_normalization_23_beta_read_readvariableop=savev2_batch_normalization_23_moving_mean_read_readvariableopAsavev2_batch_normalization_23_moving_variance_read_readvariableop*savev2_dense_95_kernel_read_readvariableop(savev2_dense_95_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_92_kernel_m_read_readvariableop/savev2_adam_dense_92_bias_m_read_readvariableop1savev2_adam_dense_93_kernel_m_read_readvariableop/savev2_adam_dense_93_bias_m_read_readvariableop1savev2_adam_dense_94_kernel_m_read_readvariableop/savev2_adam_dense_94_bias_m_read_readvariableop>savev2_adam_batch_normalization_23_gamma_m_read_readvariableop=savev2_adam_batch_normalization_23_beta_m_read_readvariableop1savev2_adam_dense_95_kernel_m_read_readvariableop/savev2_adam_dense_95_bias_m_read_readvariableop1savev2_adam_dense_92_kernel_v_read_readvariableop/savev2_adam_dense_92_bias_v_read_readvariableop1savev2_adam_dense_93_kernel_v_read_readvariableop/savev2_adam_dense_93_bias_v_read_readvariableop1savev2_adam_dense_94_kernel_v_read_readvariableop/savev2_adam_dense_94_bias_v_read_readvariableop>savev2_adam_batch_normalization_23_gamma_v_read_readvariableop=savev2_adam_batch_normalization_23_beta_v_read_readvariableop1savev2_adam_dense_95_kernel_v_read_readvariableop/savev2_adam_dense_95_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*½
_input_shapes«
¨: :
::
::
::::::	:: : : : : : : :
::
::
::::	::
::
::
::::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!	

_output_shapes	
::!


_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::&"
 
_output_shapes
:
:!

_output_shapes	
::& "
 
_output_shapes
:
:!!

_output_shapes	
::&""
 
_output_shapes
:
:!#

_output_shapes	
::!$

_output_shapes	
::!%

_output_shapes	
::%&!

_output_shapes
:	: '

_output_shapes
::(

_output_shapes
: 
Í0
Ò
U__inference_batch_normalization_23_layer_call_and_return_conditional_losses_226664095

inputs
assignmovingavg_226664070
assignmovingavg_1_226664076)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1Ï
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg/226664070*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_226664070*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpõ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg/226664070*
_output_shapes	
:2
AssignMovingAvg/subì
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg/226664070*
_output_shapes	
:2
AssignMovingAvg/mulµ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_226664070AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg/226664070*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÕ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*.
_class$
" loc:@AssignMovingAvg_1/226664076*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_226664076*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÿ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/226664076*
_output_shapes	
:2
AssignMovingAvg_1/subö
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/226664076*
_output_shapes	
:2
AssignMovingAvg_1/mulÁ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_226664076AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*.
_class$
" loc:@AssignMovingAvg_1/226664076*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1´
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó!
ý
G__inference_model_23_layer_call_and_return_conditional_losses_226664301
input_24
dense_92_226664179
dense_92_226664181
dense_93_226664206
dense_93_226664208
dense_94_226664233
dense_94_226664235$
 batch_normalization_23_226664264$
 batch_normalization_23_226664266$
 batch_normalization_23_226664268$
 batch_normalization_23_226664270
dense_95_226664295
dense_95_226664297
identity¢.batch_normalization_23/StatefulPartitionedCall¢ dense_92/StatefulPartitionedCall¢ dense_93/StatefulPartitionedCall¢ dense_94/StatefulPartitionedCall¢ dense_95/StatefulPartitionedCallà
flatten_23/PartitionedCallPartitionedCallinput_24*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_flatten_23_layer_call_and_return_conditional_losses_2266641492
flatten_23/PartitionedCall»
 dense_92/StatefulPartitionedCallStatefulPartitionedCall#flatten_23/PartitionedCall:output:0dense_92_226664179dense_92_226664181*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_92_layer_call_and_return_conditional_losses_2266641682"
 dense_92/StatefulPartitionedCallÁ
 dense_93/StatefulPartitionedCallStatefulPartitionedCall)dense_92/StatefulPartitionedCall:output:0dense_93_226664206dense_93_226664208*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_93_layer_call_and_return_conditional_losses_2266641952"
 dense_93/StatefulPartitionedCallÁ
 dense_94/StatefulPartitionedCallStatefulPartitionedCall)dense_93/StatefulPartitionedCall:output:0dense_94_226664233dense_94_226664235*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_94_layer_call_and_return_conditional_losses_2266642222"
 dense_94/StatefulPartitionedCallÍ
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall)dense_94/StatefulPartitionedCall:output:0 batch_normalization_23_226664264 batch_normalization_23_226664266 batch_normalization_23_226664268 batch_normalization_23_226664270*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_batch_normalization_23_layer_call_and_return_conditional_losses_22666409520
.batch_normalization_23/StatefulPartitionedCallÎ
 dense_95/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0dense_95_226664295dense_95_226664297*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_95_layer_call_and_return_conditional_losses_2266642842"
 dense_95/StatefulPartitionedCallº
IdentityIdentity)dense_95/StatefulPartitionedCall:output:0/^batch_normalization_23/StatefulPartitionedCall!^dense_92/StatefulPartitionedCall!^dense_93/StatefulPartitionedCall!^dense_94/StatefulPartitionedCall!^dense_95/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2D
 dense_92/StatefulPartitionedCall dense_92/StatefulPartitionedCall2D
 dense_93/StatefulPartitionedCall dense_93/StatefulPartitionedCall2D
 dense_94/StatefulPartitionedCall dense_94/StatefulPartitionedCall2D
 dense_95/StatefulPartitionedCall dense_95/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
input_24
ç

,__inference_dense_92_layer_call_fn_226664706

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_92_layer_call_and_return_conditional_losses_2266641682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±	

,__inference_model_23_layer_call_fn_226664399
input_24
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinput_24unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_model_23_layer_call_and_return_conditional_losses_2266643722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
input_24
«	

,__inference_model_23_layer_call_fn_226664646

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_model_23_layer_call_and_return_conditional_losses_2266643722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ú	
à
G__inference_dense_93_layer_call_and_return_conditional_losses_226664717

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú	
à
G__inference_dense_92_layer_call_and_return_conditional_losses_226664697

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú	
à
G__inference_dense_92_layer_call_and_return_conditional_losses_226664168

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç

,__inference_dense_93_layer_call_fn_226664726

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_93_layer_call_and_return_conditional_losses_2266641952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³	

,__inference_model_23_layer_call_fn_226664462
input_24
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinput_24unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_model_23_layer_call_and_return_conditional_losses_2266644352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
input_24
ç

,__inference_dense_94_layer_call_fn_226664746

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_94_layer_call_and_return_conditional_losses_2266642222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ðn
¢

G__inference_model_23_layer_call_and_return_conditional_losses_226664567

inputs+
'dense_92_matmul_readvariableop_resource,
(dense_92_biasadd_readvariableop_resource+
'dense_93_matmul_readvariableop_resource,
(dense_93_biasadd_readvariableop_resource+
'dense_94_matmul_readvariableop_resource,
(dense_94_biasadd_readvariableop_resource4
0batch_normalization_23_assignmovingavg_2266645356
2batch_normalization_23_assignmovingavg_1_226664541@
<batch_normalization_23_batchnorm_mul_readvariableop_resource<
8batch_normalization_23_batchnorm_readvariableop_resource+
'dense_95_matmul_readvariableop_resource,
(dense_95_biasadd_readvariableop_resource
identity¢:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_23/AssignMovingAvg/ReadVariableOp¢<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_23/batchnorm/ReadVariableOp¢3batch_normalization_23/batchnorm/mul/ReadVariableOp¢dense_92/BiasAdd/ReadVariableOp¢dense_92/MatMul/ReadVariableOp¢dense_93/BiasAdd/ReadVariableOp¢dense_93/MatMul/ReadVariableOp¢dense_94/BiasAdd/ReadVariableOp¢dense_94/MatMul/ReadVariableOp¢dense_95/BiasAdd/ReadVariableOp¢dense_95/MatMul/ReadVariableOpu
flatten_23/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_23/Const
flatten_23/ReshapeReshapeinputsflatten_23/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_23/Reshapeª
dense_92/MatMul/ReadVariableOpReadVariableOp'dense_92_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_92/MatMul/ReadVariableOp¤
dense_92/MatMulMatMulflatten_23/Reshape:output:0&dense_92/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_92/MatMul¨
dense_92/BiasAdd/ReadVariableOpReadVariableOp(dense_92_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_92/BiasAdd/ReadVariableOp¦
dense_92/BiasAddBiasAdddense_92/MatMul:product:0'dense_92/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_92/BiasAddt
dense_92/ReluReludense_92/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_92/Reluª
dense_93/MatMul/ReadVariableOpReadVariableOp'dense_93_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_93/MatMul/ReadVariableOp¤
dense_93/MatMulMatMuldense_92/Relu:activations:0&dense_93/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_93/MatMul¨
dense_93/BiasAdd/ReadVariableOpReadVariableOp(dense_93_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_93/BiasAdd/ReadVariableOp¦
dense_93/BiasAddBiasAdddense_93/MatMul:product:0'dense_93/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_93/BiasAddt
dense_93/ReluReludense_93/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_93/Reluª
dense_94/MatMul/ReadVariableOpReadVariableOp'dense_94_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_94/MatMul/ReadVariableOp¤
dense_94/MatMulMatMuldense_93/Relu:activations:0&dense_94/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_94/MatMul¨
dense_94/BiasAdd/ReadVariableOpReadVariableOp(dense_94_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_94/BiasAdd/ReadVariableOp¦
dense_94/BiasAddBiasAdddense_94/MatMul:product:0'dense_94/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_94/BiasAddt
dense_94/ReluReludense_94/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_94/Relu¸
5batch_normalization_23/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_23/moments/mean/reduction_indicesê
#batch_normalization_23/moments/meanMeandense_94/Relu:activations:0>batch_normalization_23/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2%
#batch_normalization_23/moments/meanÂ
+batch_normalization_23/moments/StopGradientStopGradient,batch_normalization_23/moments/mean:output:0*
T0*
_output_shapes
:	2-
+batch_normalization_23/moments/StopGradientÿ
0batch_normalization_23/moments/SquaredDifferenceSquaredDifferencedense_94/Relu:activations:04batch_normalization_23/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_23/moments/SquaredDifferenceÀ
9batch_normalization_23/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_23/moments/variance/reduction_indices
'batch_normalization_23/moments/varianceMean4batch_normalization_23/moments/SquaredDifference:z:0Bbatch_normalization_23/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2)
'batch_normalization_23/moments/varianceÆ
&batch_normalization_23/moments/SqueezeSqueeze,batch_normalization_23/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2(
&batch_normalization_23/moments/SqueezeÎ
(batch_normalization_23/moments/Squeeze_1Squeeze0batch_normalization_23/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2*
(batch_normalization_23/moments/Squeeze_1
,batch_normalization_23/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_23/AssignMovingAvg/226664535*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_23/AssignMovingAvg/decayÜ
5batch_normalization_23/AssignMovingAvg/ReadVariableOpReadVariableOp0batch_normalization_23_assignmovingavg_226664535*
_output_shapes	
:*
dtype027
5batch_normalization_23/AssignMovingAvg/ReadVariableOpè
*batch_normalization_23/AssignMovingAvg/subSub=batch_normalization_23/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_23/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_23/AssignMovingAvg/226664535*
_output_shapes	
:2,
*batch_normalization_23/AssignMovingAvg/subß
*batch_normalization_23/AssignMovingAvg/mulMul.batch_normalization_23/AssignMovingAvg/sub:z:05batch_normalization_23/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_23/AssignMovingAvg/226664535*
_output_shapes	
:2,
*batch_normalization_23/AssignMovingAvg/mul¿
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp0batch_normalization_23_assignmovingavg_226664535.batch_normalization_23/AssignMovingAvg/mul:z:06^batch_normalization_23/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_23/AssignMovingAvg/226664535*
_output_shapes
 *
dtype02<
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_23/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*E
_class;
97loc:@batch_normalization_23/AssignMovingAvg_1/226664541*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_23/AssignMovingAvg_1/decayâ
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOpReadVariableOp2batch_normalization_23_assignmovingavg_1_226664541*
_output_shapes	
:*
dtype029
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOpò
,batch_normalization_23/AssignMovingAvg_1/subSub?batch_normalization_23/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_23/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@batch_normalization_23/AssignMovingAvg_1/226664541*
_output_shapes	
:2.
,batch_normalization_23/AssignMovingAvg_1/subé
,batch_normalization_23/AssignMovingAvg_1/mulMul0batch_normalization_23/AssignMovingAvg_1/sub:z:07batch_normalization_23/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@batch_normalization_23/AssignMovingAvg_1/226664541*
_output_shapes	
:2.
,batch_normalization_23/AssignMovingAvg_1/mulË
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp2batch_normalization_23_assignmovingavg_1_2266645410batch_normalization_23/AssignMovingAvg_1/mul:z:08^batch_normalization_23/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*E
_class;
97loc:@batch_normalization_23/AssignMovingAvg_1/226664541*
_output_shapes
 *
dtype02>
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_23/batchnorm/add/yß
$batch_normalization_23/batchnorm/addAddV21batch_normalization_23/moments/Squeeze_1:output:0/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2&
$batch_normalization_23/batchnorm/add©
&batch_normalization_23/batchnorm/RsqrtRsqrt(batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_23/batchnorm/Rsqrtä
3batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype025
3batch_normalization_23/batchnorm/mul/ReadVariableOpâ
$batch_normalization_23/batchnorm/mulMul*batch_normalization_23/batchnorm/Rsqrt:y:0;batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2&
$batch_normalization_23/batchnorm/mulÑ
&batch_normalization_23/batchnorm/mul_1Muldense_94/Relu:activations:0(batch_normalization_23/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_23/batchnorm/mul_1Ø
&batch_normalization_23/batchnorm/mul_2Mul/batch_normalization_23/moments/Squeeze:output:0(batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_23/batchnorm/mul_2Ø
/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype021
/batch_normalization_23/batchnorm/ReadVariableOpÞ
$batch_normalization_23/batchnorm/subSub7batch_normalization_23/batchnorm/ReadVariableOp:value:0*batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2&
$batch_normalization_23/batchnorm/subâ
&batch_normalization_23/batchnorm/add_1AddV2*batch_normalization_23/batchnorm/mul_1:z:0(batch_normalization_23/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_23/batchnorm/add_1©
dense_95/MatMul/ReadVariableOpReadVariableOp'dense_95_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_95/MatMul/ReadVariableOp²
dense_95/MatMulMatMul*batch_normalization_23/batchnorm/add_1:z:0&dense_95/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_95/MatMul§
dense_95/BiasAdd/ReadVariableOpReadVariableOp(dense_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_95/BiasAdd/ReadVariableOp¥
dense_95/BiasAddBiasAdddense_95/MatMul:product:0'dense_95/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_95/BiasAdd|
dense_95/SoftmaxSoftmaxdense_95/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_95/SoftmaxÐ
IdentityIdentitydense_95/Softmax:softmax:0;^batch_normalization_23/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_23/AssignMovingAvg/ReadVariableOp=^batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_23/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_23/batchnorm/ReadVariableOp4^batch_normalization_23/batchnorm/mul/ReadVariableOp ^dense_92/BiasAdd/ReadVariableOp^dense_92/MatMul/ReadVariableOp ^dense_93/BiasAdd/ReadVariableOp^dense_93/MatMul/ReadVariableOp ^dense_94/BiasAdd/ReadVariableOp^dense_94/MatMul/ReadVariableOp ^dense_95/BiasAdd/ReadVariableOp^dense_95/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2x
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_23/AssignMovingAvg/ReadVariableOp5batch_normalization_23/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_23/batchnorm/ReadVariableOp/batch_normalization_23/batchnorm/ReadVariableOp2j
3batch_normalization_23/batchnorm/mul/ReadVariableOp3batch_normalization_23/batchnorm/mul/ReadVariableOp2B
dense_92/BiasAdd/ReadVariableOpdense_92/BiasAdd/ReadVariableOp2@
dense_92/MatMul/ReadVariableOpdense_92/MatMul/ReadVariableOp2B
dense_93/BiasAdd/ReadVariableOpdense_93/BiasAdd/ReadVariableOp2@
dense_93/MatMul/ReadVariableOpdense_93/MatMul/ReadVariableOp2B
dense_94/BiasAdd/ReadVariableOpdense_94/BiasAdd/ReadVariableOp2@
dense_94/MatMul/ReadVariableOpdense_94/MatMul/ReadVariableOp2B
dense_95/BiasAdd/ReadVariableOpdense_95/BiasAdd/ReadVariableOp2@
dense_95/MatMul/ReadVariableOpdense_95/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
æ

U__inference_batch_normalization_23_layer_call_and_return_conditional_losses_226664128

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Ü
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì¦

%__inference__traced_restore_226665115
file_prefix$
 assignvariableop_dense_92_kernel$
 assignvariableop_1_dense_92_bias&
"assignvariableop_2_dense_93_kernel$
 assignvariableop_3_dense_93_bias&
"assignvariableop_4_dense_94_kernel$
 assignvariableop_5_dense_94_bias3
/assignvariableop_6_batch_normalization_23_gamma2
.assignvariableop_7_batch_normalization_23_beta9
5assignvariableop_8_batch_normalization_23_moving_mean=
9assignvariableop_9_batch_normalization_23_moving_variance'
#assignvariableop_10_dense_95_kernel%
!assignvariableop_11_dense_95_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count.
*assignvariableop_19_adam_dense_92_kernel_m,
(assignvariableop_20_adam_dense_92_bias_m.
*assignvariableop_21_adam_dense_93_kernel_m,
(assignvariableop_22_adam_dense_93_bias_m.
*assignvariableop_23_adam_dense_94_kernel_m,
(assignvariableop_24_adam_dense_94_bias_m;
7assignvariableop_25_adam_batch_normalization_23_gamma_m:
6assignvariableop_26_adam_batch_normalization_23_beta_m.
*assignvariableop_27_adam_dense_95_kernel_m,
(assignvariableop_28_adam_dense_95_bias_m.
*assignvariableop_29_adam_dense_92_kernel_v,
(assignvariableop_30_adam_dense_92_bias_v.
*assignvariableop_31_adam_dense_93_kernel_v,
(assignvariableop_32_adam_dense_93_bias_v.
*assignvariableop_33_adam_dense_94_kernel_v,
(assignvariableop_34_adam_dense_94_bias_v;
7assignvariableop_35_adam_batch_normalization_23_gamma_v:
6assignvariableop_36_adam_batch_normalization_23_beta_v.
*assignvariableop_37_adam_dense_95_kernel_v,
(assignvariableop_38_adam_dense_95_bias_v
identity_40¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*¡
valueB(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÞ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesö
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¶
_output_shapes£
 ::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_92_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_92_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_93_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_93_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_94_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_94_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6´
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_23_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7³
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_23_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8º
AssignVariableOp_8AssignVariableOp5assignvariableop_8_batch_normalization_23_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¾
AssignVariableOp_9AssignVariableOp9assignvariableop_9_batch_normalization_23_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10«
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_95_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11©
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_95_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12¥
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13§
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14§
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¦
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16®
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¡
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¡
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19²
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_92_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20°
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_92_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21²
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_93_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22°
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_93_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23²
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_94_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24°
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_94_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¿
AssignVariableOp_25AssignVariableOp7assignvariableop_25_adam_batch_normalization_23_gamma_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¾
AssignVariableOp_26AssignVariableOp6assignvariableop_26_adam_batch_normalization_23_beta_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27²
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_95_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28°
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_95_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29²
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_92_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30°
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_92_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31²
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_93_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32°
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_93_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33²
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_94_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34°
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_94_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¿
AssignVariableOp_35AssignVariableOp7assignvariableop_35_adam_batch_normalization_23_gamma_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¾
AssignVariableOp_36AssignVariableOp6assignvariableop_36_adam_batch_normalization_23_beta_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37²
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_95_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38°
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_95_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¸
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39«
Identity_40IdentityIdentity_39:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_40"#
identity_40Identity_40:output:0*³
_input_shapes¡
: :::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ü	
à
G__inference_dense_95_layer_call_and_return_conditional_losses_226664839

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú	
à
G__inference_dense_94_layer_call_and_return_conditional_losses_226664222

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹R
å

$__inference__wrapped_model_226663999
input_244
0model_23_dense_92_matmul_readvariableop_resource5
1model_23_dense_92_biasadd_readvariableop_resource4
0model_23_dense_93_matmul_readvariableop_resource5
1model_23_dense_93_biasadd_readvariableop_resource4
0model_23_dense_94_matmul_readvariableop_resource5
1model_23_dense_94_biasadd_readvariableop_resourceE
Amodel_23_batch_normalization_23_batchnorm_readvariableop_resourceI
Emodel_23_batch_normalization_23_batchnorm_mul_readvariableop_resourceG
Cmodel_23_batch_normalization_23_batchnorm_readvariableop_1_resourceG
Cmodel_23_batch_normalization_23_batchnorm_readvariableop_2_resource4
0model_23_dense_95_matmul_readvariableop_resource5
1model_23_dense_95_biasadd_readvariableop_resource
identity¢8model_23/batch_normalization_23/batchnorm/ReadVariableOp¢:model_23/batch_normalization_23/batchnorm/ReadVariableOp_1¢:model_23/batch_normalization_23/batchnorm/ReadVariableOp_2¢<model_23/batch_normalization_23/batchnorm/mul/ReadVariableOp¢(model_23/dense_92/BiasAdd/ReadVariableOp¢'model_23/dense_92/MatMul/ReadVariableOp¢(model_23/dense_93/BiasAdd/ReadVariableOp¢'model_23/dense_93/MatMul/ReadVariableOp¢(model_23/dense_94/BiasAdd/ReadVariableOp¢'model_23/dense_94/MatMul/ReadVariableOp¢(model_23/dense_95/BiasAdd/ReadVariableOp¢'model_23/dense_95/MatMul/ReadVariableOp
model_23/flatten_23/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model_23/flatten_23/Const¦
model_23/flatten_23/ReshapeReshapeinput_24"model_23/flatten_23/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_23/flatten_23/ReshapeÅ
'model_23/dense_92/MatMul/ReadVariableOpReadVariableOp0model_23_dense_92_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'model_23/dense_92/MatMul/ReadVariableOpÈ
model_23/dense_92/MatMulMatMul$model_23/flatten_23/Reshape:output:0/model_23/dense_92/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_23/dense_92/MatMulÃ
(model_23/dense_92/BiasAdd/ReadVariableOpReadVariableOp1model_23_dense_92_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_23/dense_92/BiasAdd/ReadVariableOpÊ
model_23/dense_92/BiasAddBiasAdd"model_23/dense_92/MatMul:product:00model_23/dense_92/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_23/dense_92/BiasAdd
model_23/dense_92/ReluRelu"model_23/dense_92/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_23/dense_92/ReluÅ
'model_23/dense_93/MatMul/ReadVariableOpReadVariableOp0model_23_dense_93_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'model_23/dense_93/MatMul/ReadVariableOpÈ
model_23/dense_93/MatMulMatMul$model_23/dense_92/Relu:activations:0/model_23/dense_93/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_23/dense_93/MatMulÃ
(model_23/dense_93/BiasAdd/ReadVariableOpReadVariableOp1model_23_dense_93_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_23/dense_93/BiasAdd/ReadVariableOpÊ
model_23/dense_93/BiasAddBiasAdd"model_23/dense_93/MatMul:product:00model_23/dense_93/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_23/dense_93/BiasAdd
model_23/dense_93/ReluRelu"model_23/dense_93/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_23/dense_93/ReluÅ
'model_23/dense_94/MatMul/ReadVariableOpReadVariableOp0model_23_dense_94_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02)
'model_23/dense_94/MatMul/ReadVariableOpÈ
model_23/dense_94/MatMulMatMul$model_23/dense_93/Relu:activations:0/model_23/dense_94/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_23/dense_94/MatMulÃ
(model_23/dense_94/BiasAdd/ReadVariableOpReadVariableOp1model_23_dense_94_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_23/dense_94/BiasAdd/ReadVariableOpÊ
model_23/dense_94/BiasAddBiasAdd"model_23/dense_94/MatMul:product:00model_23/dense_94/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_23/dense_94/BiasAdd
model_23/dense_94/ReluRelu"model_23/dense_94/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_23/dense_94/Reluó
8model_23/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOpAmodel_23_batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02:
8model_23/batch_normalization_23/batchnorm/ReadVariableOp§
/model_23/batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:21
/model_23/batch_normalization_23/batchnorm/add/y
-model_23/batch_normalization_23/batchnorm/addAddV2@model_23/batch_normalization_23/batchnorm/ReadVariableOp:value:08model_23/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2/
-model_23/batch_normalization_23/batchnorm/addÄ
/model_23/batch_normalization_23/batchnorm/RsqrtRsqrt1model_23/batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes	
:21
/model_23/batch_normalization_23/batchnorm/Rsqrtÿ
<model_23/batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOpEmodel_23_batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02>
<model_23/batch_normalization_23/batchnorm/mul/ReadVariableOp
-model_23/batch_normalization_23/batchnorm/mulMul3model_23/batch_normalization_23/batchnorm/Rsqrt:y:0Dmodel_23/batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2/
-model_23/batch_normalization_23/batchnorm/mulõ
/model_23/batch_normalization_23/batchnorm/mul_1Mul$model_23/dense_94/Relu:activations:01model_23/batch_normalization_23/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/model_23/batch_normalization_23/batchnorm/mul_1ù
:model_23/batch_normalization_23/batchnorm/ReadVariableOp_1ReadVariableOpCmodel_23_batch_normalization_23_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02<
:model_23/batch_normalization_23/batchnorm/ReadVariableOp_1
/model_23/batch_normalization_23/batchnorm/mul_2MulBmodel_23/batch_normalization_23/batchnorm/ReadVariableOp_1:value:01model_23/batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes	
:21
/model_23/batch_normalization_23/batchnorm/mul_2ù
:model_23/batch_normalization_23/batchnorm/ReadVariableOp_2ReadVariableOpCmodel_23_batch_normalization_23_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02<
:model_23/batch_normalization_23/batchnorm/ReadVariableOp_2
-model_23/batch_normalization_23/batchnorm/subSubBmodel_23/batch_normalization_23/batchnorm/ReadVariableOp_2:value:03model_23/batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2/
-model_23/batch_normalization_23/batchnorm/sub
/model_23/batch_normalization_23/batchnorm/add_1AddV23model_23/batch_normalization_23/batchnorm/mul_1:z:01model_23/batch_normalization_23/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/model_23/batch_normalization_23/batchnorm/add_1Ä
'model_23/dense_95/MatMul/ReadVariableOpReadVariableOp0model_23_dense_95_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02)
'model_23/dense_95/MatMul/ReadVariableOpÖ
model_23/dense_95/MatMulMatMul3model_23/batch_normalization_23/batchnorm/add_1:z:0/model_23/dense_95/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_23/dense_95/MatMulÂ
(model_23/dense_95/BiasAdd/ReadVariableOpReadVariableOp1model_23_dense_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_23/dense_95/BiasAdd/ReadVariableOpÉ
model_23/dense_95/BiasAddBiasAdd"model_23/dense_95/MatMul:product:00model_23/dense_95/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_23/dense_95/BiasAdd
model_23/dense_95/SoftmaxSoftmax"model_23/dense_95/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_23/dense_95/Softmax¿
IdentityIdentity#model_23/dense_95/Softmax:softmax:09^model_23/batch_normalization_23/batchnorm/ReadVariableOp;^model_23/batch_normalization_23/batchnorm/ReadVariableOp_1;^model_23/batch_normalization_23/batchnorm/ReadVariableOp_2=^model_23/batch_normalization_23/batchnorm/mul/ReadVariableOp)^model_23/dense_92/BiasAdd/ReadVariableOp(^model_23/dense_92/MatMul/ReadVariableOp)^model_23/dense_93/BiasAdd/ReadVariableOp(^model_23/dense_93/MatMul/ReadVariableOp)^model_23/dense_94/BiasAdd/ReadVariableOp(^model_23/dense_94/MatMul/ReadVariableOp)^model_23/dense_95/BiasAdd/ReadVariableOp(^model_23/dense_95/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2t
8model_23/batch_normalization_23/batchnorm/ReadVariableOp8model_23/batch_normalization_23/batchnorm/ReadVariableOp2x
:model_23/batch_normalization_23/batchnorm/ReadVariableOp_1:model_23/batch_normalization_23/batchnorm/ReadVariableOp_12x
:model_23/batch_normalization_23/batchnorm/ReadVariableOp_2:model_23/batch_normalization_23/batchnorm/ReadVariableOp_22|
<model_23/batch_normalization_23/batchnorm/mul/ReadVariableOp<model_23/batch_normalization_23/batchnorm/mul/ReadVariableOp2T
(model_23/dense_92/BiasAdd/ReadVariableOp(model_23/dense_92/BiasAdd/ReadVariableOp2R
'model_23/dense_92/MatMul/ReadVariableOp'model_23/dense_92/MatMul/ReadVariableOp2T
(model_23/dense_93/BiasAdd/ReadVariableOp(model_23/dense_93/BiasAdd/ReadVariableOp2R
'model_23/dense_93/MatMul/ReadVariableOp'model_23/dense_93/MatMul/ReadVariableOp2T
(model_23/dense_94/BiasAdd/ReadVariableOp(model_23/dense_94/BiasAdd/ReadVariableOp2R
'model_23/dense_94/MatMul/ReadVariableOp'model_23/dense_94/MatMul/ReadVariableOp2T
(model_23/dense_95/BiasAdd/ReadVariableOp(model_23/dense_95/BiasAdd/ReadVariableOp2R
'model_23/dense_95/MatMul/ReadVariableOp'model_23/dense_95/MatMul/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
input_24
¹
e
I__inference_flatten_23_layer_call_and_return_conditional_losses_226664681

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ4:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
Ã
­
:__inference_batch_normalization_23_layer_call_fn_226664828

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_batch_normalization_23_layer_call_and_return_conditional_losses_2266641282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å

,__inference_dense_95_layer_call_fn_226664848

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_95_layer_call_and_return_conditional_losses_2266642842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
e
I__inference_flatten_23_layer_call_and_return_conditional_losses_226664149

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ4:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
Ï!
û
G__inference_model_23_layer_call_and_return_conditional_losses_226664435

inputs
dense_92_226664405
dense_92_226664407
dense_93_226664410
dense_93_226664412
dense_94_226664415
dense_94_226664417$
 batch_normalization_23_226664420$
 batch_normalization_23_226664422$
 batch_normalization_23_226664424$
 batch_normalization_23_226664426
dense_95_226664429
dense_95_226664431
identity¢.batch_normalization_23/StatefulPartitionedCall¢ dense_92/StatefulPartitionedCall¢ dense_93/StatefulPartitionedCall¢ dense_94/StatefulPartitionedCall¢ dense_95/StatefulPartitionedCallÞ
flatten_23/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_flatten_23_layer_call_and_return_conditional_losses_2266641492
flatten_23/PartitionedCall»
 dense_92/StatefulPartitionedCallStatefulPartitionedCall#flatten_23/PartitionedCall:output:0dense_92_226664405dense_92_226664407*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_92_layer_call_and_return_conditional_losses_2266641682"
 dense_92/StatefulPartitionedCallÁ
 dense_93/StatefulPartitionedCallStatefulPartitionedCall)dense_92/StatefulPartitionedCall:output:0dense_93_226664410dense_93_226664412*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_93_layer_call_and_return_conditional_losses_2266641952"
 dense_93/StatefulPartitionedCallÁ
 dense_94/StatefulPartitionedCallStatefulPartitionedCall)dense_93/StatefulPartitionedCall:output:0dense_94_226664415dense_94_226664417*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_94_layer_call_and_return_conditional_losses_2266642222"
 dense_94/StatefulPartitionedCallÏ
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall)dense_94/StatefulPartitionedCall:output:0 batch_normalization_23_226664420 batch_normalization_23_226664422 batch_normalization_23_226664424 batch_normalization_23_226664426*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_batch_normalization_23_layer_call_and_return_conditional_losses_22666412820
.batch_normalization_23/StatefulPartitionedCallÎ
 dense_95/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0dense_95_226664429dense_95_226664431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_95_layer_call_and_return_conditional_losses_2266642842"
 dense_95/StatefulPartitionedCallº
IdentityIdentity)dense_95/StatefulPartitionedCall:output:0/^batch_normalization_23/StatefulPartitionedCall!^dense_92/StatefulPartitionedCall!^dense_93/StatefulPartitionedCall!^dense_94/StatefulPartitionedCall!^dense_95/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2D
 dense_92/StatefulPartitionedCall dense_92/StatefulPartitionedCall2D
 dense_93/StatefulPartitionedCall dense_93/StatefulPartitionedCall2D
 dense_94/StatefulPartitionedCall dense_94/StatefulPartitionedCall2D
 dense_95/StatefulPartitionedCall dense_95/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ú	
à
G__inference_dense_93_layer_call_and_return_conditional_losses_226664195

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­	

,__inference_model_23_layer_call_fn_226664675

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_model_23_layer_call_and_return_conditional_losses_2266644352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ú	
à
G__inference_dense_94_layer_call_and_return_conditional_losses_226664737

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ

U__inference_batch_normalization_23_layer_call_and_return_conditional_losses_226664802

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1Ü
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í0
Ò
U__inference_batch_normalization_23_layer_call_and_return_conditional_losses_226664782

inputs
assignmovingavg_226664757
assignmovingavg_1_226664763)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1Ï
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg/226664757*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_226664757*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpõ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg/226664757*
_output_shapes	
:2
AssignMovingAvg/subì
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg/226664757*
_output_shapes	
:2
AssignMovingAvg/mulµ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_226664757AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg/226664757*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÕ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*.
_class$
" loc:@AssignMovingAvg_1/226664763*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_226664763*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÿ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/226664763*
_output_shapes	
:2
AssignMovingAvg_1/subö
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/226664763*
_output_shapes	
:2
AssignMovingAvg_1/mulÁ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_226664763AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*.
_class$
" loc:@AssignMovingAvg_1/226664763*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1´
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ!
ý
G__inference_model_23_layer_call_and_return_conditional_losses_226664335
input_24
dense_92_226664305
dense_92_226664307
dense_93_226664310
dense_93_226664312
dense_94_226664315
dense_94_226664317$
 batch_normalization_23_226664320$
 batch_normalization_23_226664322$
 batch_normalization_23_226664324$
 batch_normalization_23_226664326
dense_95_226664329
dense_95_226664331
identity¢.batch_normalization_23/StatefulPartitionedCall¢ dense_92/StatefulPartitionedCall¢ dense_93/StatefulPartitionedCall¢ dense_94/StatefulPartitionedCall¢ dense_95/StatefulPartitionedCallà
flatten_23/PartitionedCallPartitionedCallinput_24*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_flatten_23_layer_call_and_return_conditional_losses_2266641492
flatten_23/PartitionedCall»
 dense_92/StatefulPartitionedCallStatefulPartitionedCall#flatten_23/PartitionedCall:output:0dense_92_226664305dense_92_226664307*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_92_layer_call_and_return_conditional_losses_2266641682"
 dense_92/StatefulPartitionedCallÁ
 dense_93/StatefulPartitionedCallStatefulPartitionedCall)dense_92/StatefulPartitionedCall:output:0dense_93_226664310dense_93_226664312*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_93_layer_call_and_return_conditional_losses_2266641952"
 dense_93/StatefulPartitionedCallÁ
 dense_94/StatefulPartitionedCallStatefulPartitionedCall)dense_93/StatefulPartitionedCall:output:0dense_94_226664315dense_94_226664317*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_94_layer_call_and_return_conditional_losses_2266642222"
 dense_94/StatefulPartitionedCallÏ
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall)dense_94/StatefulPartitionedCall:output:0 batch_normalization_23_226664320 batch_normalization_23_226664322 batch_normalization_23_226664324 batch_normalization_23_226664326*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_batch_normalization_23_layer_call_and_return_conditional_losses_22666412820
.batch_normalization_23/StatefulPartitionedCallÎ
 dense_95/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0dense_95_226664329dense_95_226664331*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_95_layer_call_and_return_conditional_losses_2266642842"
 dense_95/StatefulPartitionedCallº
IdentityIdentity)dense_95/StatefulPartitionedCall:output:0/^batch_normalization_23/StatefulPartitionedCall!^dense_92/StatefulPartitionedCall!^dense_93/StatefulPartitionedCall!^dense_94/StatefulPartitionedCall!^dense_95/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2D
 dense_92/StatefulPartitionedCall dense_92/StatefulPartitionedCall2D
 dense_93/StatefulPartitionedCall dense_93/StatefulPartitionedCall2D
 dense_94/StatefulPartitionedCall dense_94/StatefulPartitionedCall2D
 dense_95/StatefulPartitionedCall dense_95/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
input_24"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*±
serving_default
A
input_245
serving_default_input_24:0ÿÿÿÿÿÿÿÿÿ4<
dense_950
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:óâ
ã=
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
*t&call_and_return_all_conditional_losses
u_default_save_signature
v__call__"»:
_tf_keras_network:{"class_name": "Functional", "name": "model_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_23", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_24"}, "name": "input_24", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_23", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_23", "inbound_nodes": [[["input_24", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_92", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_92", "inbound_nodes": [[["flatten_23", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_93", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_93", "inbound_nodes": [[["dense_92", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_94", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_94", "inbound_nodes": [[["dense_93", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_23", "inbound_nodes": [[["dense_94", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_95", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_95", "inbound_nodes": [[["batch_normalization_23", 0, 0, {}]]]}], "input_layers": [["input_24", 0, 0]], "output_layers": [["dense_95", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 52, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_23", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_24"}, "name": "input_24", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_23", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_23", "inbound_nodes": [[["input_24", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_92", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_92", "inbound_nodes": [[["flatten_23", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_93", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_93", "inbound_nodes": [[["dense_92", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_94", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_94", "inbound_nodes": [[["dense_93", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_23", "inbound_nodes": [[["dense_94", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_95", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_95", "inbound_nodes": [[["batch_normalization_23", 0, 0, {}]]]}], "input_layers": [["input_24", 0, 0]], "output_layers": [["dense_95", 0, 0]]}}, "training_config": {"loss": "loss_func", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "clipnorm": 1, "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ó"ð
_tf_keras_input_layerÐ{"class_name": "InputLayer", "name": "input_24", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_24"}}
è
regularization_losses
	variables
trainable_variables
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"Ù
_tf_keras_layer¿{"class_name": "Flatten", "name": "flatten_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_23", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*y&call_and_return_all_conditional_losses
z__call__"ñ
_tf_keras_layer×{"class_name": "Dense", "name": "dense_92", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_92", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 156}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 156]}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*{&call_and_return_all_conditional_losses
|__call__"ñ
_tf_keras_layer×{"class_name": "Dense", "name": "dense_93", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_93", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}


kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
*}&call_and_return_all_conditional_losses
~__call__"ñ
_tf_keras_layer×{"class_name": "Dense", "name": "dense_94", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_94", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
·	
$axis
	%gamma
&beta
'moving_mean
(moving_variance
)regularization_losses
*	variables
+trainable_variables
,	keras_api
*&call_and_return_all_conditional_losses
__call__"â
_tf_keras_layerÈ{"class_name": "BatchNormalization", "name": "batch_normalization_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ø

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+&call_and_return_all_conditional_losses
__call__"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_95", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_95", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}

3iter

4beta_1

5beta_2
	6decay
7learning_ratem`mambmcmdme%mf&mg-mh.mivjvkvlvmvnvo%vp&vq-vr.vs"
	optimizer
 "
trackable_list_wrapper
v
0
1
2
3
4
5
%6
&7
'8
(9
-10
.11"
trackable_list_wrapper
f
0
1
2
3
4
5
%6
&7
-8
.9"
trackable_list_wrapper
Ê

8layers
9metrics
	regularization_losses
:non_trainable_variables

	variables
;layer_regularization_losses
<layer_metrics
trainable_variables
v__call__
u_default_save_signature
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
=metrics

>layers
regularization_losses
?non_trainable_variables
	variables
@layer_regularization_losses
Alayer_metrics
trainable_variables
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_92/kernel
:2dense_92/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Bmetrics

Clayers
regularization_losses
Dnon_trainable_variables
	variables
Elayer_regularization_losses
Flayer_metrics
trainable_variables
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_93/kernel
:2dense_93/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Gmetrics

Hlayers
regularization_losses
Inon_trainable_variables
	variables
Jlayer_regularization_losses
Klayer_metrics
trainable_variables
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_94/kernel
:2dense_94/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Lmetrics

Mlayers
 regularization_losses
Nnon_trainable_variables
!	variables
Olayer_regularization_losses
Player_metrics
"trainable_variables
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_23/gamma
*:(2batch_normalization_23/beta
3:1 (2"batch_normalization_23/moving_mean
7:5 (2&batch_normalization_23/moving_variance
 "
trackable_list_wrapper
<
%0
&1
'2
(3"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
®
Qmetrics

Rlayers
)regularization_losses
Snon_trainable_variables
*	variables
Tlayer_regularization_losses
Ulayer_metrics
+trainable_variables
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
": 	2dense_95/kernel
:2dense_95/bias
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
°
Vmetrics

Wlayers
/regularization_losses
Xnon_trainable_variables
0	variables
Ylayer_regularization_losses
Zlayer_metrics
1trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
'
[0"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
»
	\total
	]count
^	variables
_	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
\0
]1"
trackable_list_wrapper
-
^	variables"
_generic_user_object
(:&
2Adam/dense_92/kernel/m
!:2Adam/dense_92/bias/m
(:&
2Adam/dense_93/kernel/m
!:2Adam/dense_93/bias/m
(:&
2Adam/dense_94/kernel/m
!:2Adam/dense_94/bias/m
0:.2#Adam/batch_normalization_23/gamma/m
/:-2"Adam/batch_normalization_23/beta/m
':%	2Adam/dense_95/kernel/m
 :2Adam/dense_95/bias/m
(:&
2Adam/dense_92/kernel/v
!:2Adam/dense_92/bias/v
(:&
2Adam/dense_93/kernel/v
!:2Adam/dense_93/bias/v
(:&
2Adam/dense_94/kernel/v
!:2Adam/dense_94/bias/v
0:.2#Adam/batch_normalization_23/gamma/v
/:-2"Adam/batch_normalization_23/beta/v
':%	2Adam/dense_95/kernel/v
 :2Adam/dense_95/bias/v
ê2ç
G__inference_model_23_layer_call_and_return_conditional_losses_226664567
G__inference_model_23_layer_call_and_return_conditional_losses_226664301
G__inference_model_23_layer_call_and_return_conditional_losses_226664617
G__inference_model_23_layer_call_and_return_conditional_losses_226664335À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ç2ä
$__inference__wrapped_model_226663999»
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *+¢(
&#
input_24ÿÿÿÿÿÿÿÿÿ4
þ2û
,__inference_model_23_layer_call_fn_226664675
,__inference_model_23_layer_call_fn_226664462
,__inference_model_23_layer_call_fn_226664399
,__inference_model_23_layer_call_fn_226664646À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ó2ð
I__inference_flatten_23_layer_call_and_return_conditional_losses_226664681¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_flatten_23_layer_call_fn_226664686¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_92_layer_call_and_return_conditional_losses_226664697¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_92_layer_call_fn_226664706¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_93_layer_call_and_return_conditional_losses_226664717¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_93_layer_call_fn_226664726¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_94_layer_call_and_return_conditional_losses_226664737¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_94_layer_call_fn_226664746¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
è2å
U__inference_batch_normalization_23_layer_call_and_return_conditional_losses_226664802
U__inference_batch_normalization_23_layer_call_and_return_conditional_losses_226664782´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
²2¯
:__inference_batch_normalization_23_layer_call_fn_226664828
:__inference_batch_normalization_23_layer_call_fn_226664815´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ñ2î
G__inference_dense_95_layer_call_and_return_conditional_losses_226664839¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_95_layer_call_fn_226664848¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÏBÌ
'__inference_signature_wrapper_226664501input_24"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ¢
$__inference__wrapped_model_226663999z(%'&-.5¢2
+¢(
&#
input_24ÿÿÿÿÿÿÿÿÿ4
ª "3ª0
.
dense_95"
dense_95ÿÿÿÿÿÿÿÿÿ½
U__inference_batch_normalization_23_layer_call_and_return_conditional_losses_226664782d'(%&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ½
U__inference_batch_normalization_23_layer_call_and_return_conditional_losses_226664802d(%'&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
:__inference_batch_normalization_23_layer_call_fn_226664815W'(%&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
:__inference_batch_normalization_23_layer_call_fn_226664828W(%'&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_92_layer_call_and_return_conditional_losses_226664697^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_92_layer_call_fn_226664706Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_93_layer_call_and_return_conditional_losses_226664717^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_93_layer_call_fn_226664726Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_94_layer_call_and_return_conditional_losses_226664737^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_94_layer_call_fn_226664746Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
G__inference_dense_95_layer_call_and_return_conditional_losses_226664839]-.0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_95_layer_call_fn_226664848P-.0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
I__inference_flatten_23_layer_call_and_return_conditional_losses_226664681]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ4
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_flatten_23_layer_call_fn_226664686P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ4
ª "ÿÿÿÿÿÿÿÿÿ¿
G__inference_model_23_layer_call_and_return_conditional_losses_226664301t'(%&-.=¢:
3¢0
&#
input_24ÿÿÿÿÿÿÿÿÿ4
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¿
G__inference_model_23_layer_call_and_return_conditional_losses_226664335t(%'&-.=¢:
3¢0
&#
input_24ÿÿÿÿÿÿÿÿÿ4
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
G__inference_model_23_layer_call_and_return_conditional_losses_226664567r'(%&-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
G__inference_model_23_layer_call_and_return_conditional_losses_226664617r(%'&-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_model_23_layer_call_fn_226664399g'(%&-.=¢:
3¢0
&#
input_24ÿÿÿÿÿÿÿÿÿ4
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_model_23_layer_call_fn_226664462g(%'&-.=¢:
3¢0
&#
input_24ÿÿÿÿÿÿÿÿÿ4
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_model_23_layer_call_fn_226664646e'(%&-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_model_23_layer_call_fn_226664675e(%'&-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p 

 
ª "ÿÿÿÿÿÿÿÿÿ²
'__inference_signature_wrapper_226664501(%'&-.A¢>
¢ 
7ª4
2
input_24&#
input_24ÿÿÿÿÿÿÿÿÿ4"3ª0
.
dense_95"
dense_95ÿÿÿÿÿÿÿÿÿ