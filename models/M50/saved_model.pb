èø

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
 "serve*2.4.12unknown8	
~
dense_428/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_428/kernel
w
$dense_428/kernel/Read/ReadVariableOpReadVariableOpdense_428/kernel* 
_output_shapes
:
*
dtype0
u
dense_428/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_428/bias
n
"dense_428/bias/Read/ReadVariableOpReadVariableOpdense_428/bias*
_output_shapes	
:*
dtype0
~
dense_429/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_429/kernel
w
$dense_429/kernel/Read/ReadVariableOpReadVariableOpdense_429/kernel* 
_output_shapes
:
*
dtype0
u
dense_429/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_429/bias
n
"dense_429/bias/Read/ReadVariableOpReadVariableOpdense_429/bias*
_output_shapes	
:*
dtype0
~
dense_430/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_430/kernel
w
$dense_430/kernel/Read/ReadVariableOpReadVariableOpdense_430/kernel* 
_output_shapes
:
*
dtype0
u
dense_430/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_430/bias
n
"dense_430/bias/Read/ReadVariableOpReadVariableOpdense_430/bias*
_output_shapes	
:*
dtype0

batch_normalization_107/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_107/gamma

1batch_normalization_107/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_107/gamma*
_output_shapes	
:*
dtype0

batch_normalization_107/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_107/beta

0batch_normalization_107/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_107/beta*
_output_shapes	
:*
dtype0

#batch_normalization_107/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_107/moving_mean

7batch_normalization_107/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_107/moving_mean*
_output_shapes	
:*
dtype0
§
'batch_normalization_107/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_107/moving_variance
 
;batch_normalization_107/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_107/moving_variance*
_output_shapes	
:*
dtype0
}
dense_431/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_namedense_431/kernel
v
$dense_431/kernel/Read/ReadVariableOpReadVariableOpdense_431/kernel*
_output_shapes
:	*
dtype0
t
dense_431/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_431/bias
m
"dense_431/bias/Read/ReadVariableOpReadVariableOpdense_431/bias*
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

Adam/dense_428/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_428/kernel/m

+Adam/dense_428/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_428/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_428/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_428/bias/m
|
)Adam/dense_428/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_428/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_429/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_429/kernel/m

+Adam/dense_429/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_429/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_429/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_429/bias/m
|
)Adam/dense_429/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_429/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_430/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_430/kernel/m

+Adam/dense_430/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_430/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_430/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_430/bias/m
|
)Adam/dense_430/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_430/bias/m*
_output_shapes	
:*
dtype0
¡
$Adam/batch_normalization_107/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_107/gamma/m

8Adam/batch_normalization_107/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_107/gamma/m*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_107/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_107/beta/m

7Adam/batch_normalization_107/beta/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_107/beta/m*
_output_shapes	
:*
dtype0

Adam/dense_431/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/dense_431/kernel/m

+Adam/dense_431/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_431/kernel/m*
_output_shapes
:	*
dtype0

Adam/dense_431/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_431/bias/m
{
)Adam/dense_431/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_431/bias/m*
_output_shapes
:*
dtype0

Adam/dense_428/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_428/kernel/v

+Adam/dense_428/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_428/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_428/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_428/bias/v
|
)Adam/dense_428/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_428/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_429/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_429/kernel/v

+Adam/dense_429/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_429/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_429/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_429/bias/v
|
)Adam/dense_429/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_429/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_430/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_430/kernel/v

+Adam/dense_430/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_430/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_430/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_430/bias/v
|
)Adam/dense_430/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_430/bias/v*
_output_shapes	
:*
dtype0
¡
$Adam/batch_normalization_107/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/batch_normalization_107/gamma/v

8Adam/batch_normalization_107/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/batch_normalization_107/gamma/v*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_107/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_107/beta/v

7Adam/batch_normalization_107/beta/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_107/beta/v*
_output_shapes	
:*
dtype0

Adam/dense_431/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/dense_431/kernel/v

+Adam/dense_431/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_431/kernel/v*
_output_shapes
:	*
dtype0

Adam/dense_431/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_431/bias/v
{
)Adam/dense_431/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_431/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Ü:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*:
value:B: B:
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
\Z
VARIABLE_VALUEdense_428/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_428/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
\Z
VARIABLE_VALUEdense_429/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_429/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
\Z
VARIABLE_VALUEdense_430/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_430/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
hf
VARIABLE_VALUEbatch_normalization_107/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_107/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_107/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_107/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
\Z
VARIABLE_VALUEdense_431/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_431/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
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
}
VARIABLE_VALUEAdam/dense_428/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_428/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_429/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_429/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_430/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_430/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/batch_normalization_107/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_107/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_431/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_431/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_428/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_428/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_429/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_429/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_430/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_430/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/batch_normalization_107/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_107/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_431/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_431/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_108Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ4
ã
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_108dense_428/kerneldense_428/biasdense_429/kerneldense_429/biasdense_430/kerneldense_430/bias'batch_normalization_107/moving_variancebatch_normalization_107/gamma#batch_normalization_107/moving_meanbatch_normalization_107/betadense_431/kerneldense_431/bias*
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
GPU 2J 8 *1
f,R*
(__inference_signature_wrapper_1478345180
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Î
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_428/kernel/Read/ReadVariableOp"dense_428/bias/Read/ReadVariableOp$dense_429/kernel/Read/ReadVariableOp"dense_429/bias/Read/ReadVariableOp$dense_430/kernel/Read/ReadVariableOp"dense_430/bias/Read/ReadVariableOp1batch_normalization_107/gamma/Read/ReadVariableOp0batch_normalization_107/beta/Read/ReadVariableOp7batch_normalization_107/moving_mean/Read/ReadVariableOp;batch_normalization_107/moving_variance/Read/ReadVariableOp$dense_431/kernel/Read/ReadVariableOp"dense_431/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_428/kernel/m/Read/ReadVariableOp)Adam/dense_428/bias/m/Read/ReadVariableOp+Adam/dense_429/kernel/m/Read/ReadVariableOp)Adam/dense_429/bias/m/Read/ReadVariableOp+Adam/dense_430/kernel/m/Read/ReadVariableOp)Adam/dense_430/bias/m/Read/ReadVariableOp8Adam/batch_normalization_107/gamma/m/Read/ReadVariableOp7Adam/batch_normalization_107/beta/m/Read/ReadVariableOp+Adam/dense_431/kernel/m/Read/ReadVariableOp)Adam/dense_431/bias/m/Read/ReadVariableOp+Adam/dense_428/kernel/v/Read/ReadVariableOp)Adam/dense_428/bias/v/Read/ReadVariableOp+Adam/dense_429/kernel/v/Read/ReadVariableOp)Adam/dense_429/bias/v/Read/ReadVariableOp+Adam/dense_430/kernel/v/Read/ReadVariableOp)Adam/dense_430/bias/v/Read/ReadVariableOp8Adam/batch_normalization_107/gamma/v/Read/ReadVariableOp7Adam/batch_normalization_107/beta/v/Read/ReadVariableOp+Adam/dense_431/kernel/v/Read/ReadVariableOp)Adam/dense_431/bias/v/Read/ReadVariableOpConst*4
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
GPU 2J 8 *,
f'R%
#__inference__traced_save_1478345667
½	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_428/kerneldense_428/biasdense_429/kerneldense_429/biasdense_430/kerneldense_430/biasbatch_normalization_107/gammabatch_normalization_107/beta#batch_normalization_107/moving_mean'batch_normalization_107/moving_variancedense_431/kerneldense_431/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_428/kernel/mAdam/dense_428/bias/mAdam/dense_429/kernel/mAdam/dense_429/bias/mAdam/dense_430/kernel/mAdam/dense_430/bias/m$Adam/batch_normalization_107/gamma/m#Adam/batch_normalization_107/beta/mAdam/dense_431/kernel/mAdam/dense_431/bias/mAdam/dense_428/kernel/vAdam/dense_428/bias/vAdam/dense_429/kernel/vAdam/dense_429/bias/vAdam/dense_430/kernel/vAdam/dense_430/bias/v$Adam/batch_normalization_107/gamma/v#Adam/batch_normalization_107/beta/vAdam/dense_431/kernel/vAdam/dense_431/bias/v*3
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
GPU 2J 8 */
f*R(
&__inference__traced_restore_1478345794¡×
ü	
â
I__inference_dense_429_layer_call_and_return_conditional_losses_1478345396

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
þ	
â
I__inference_dense_431_layer_call_and_return_conditional_losses_1478345518

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
º"

I__inference_model_107_layer_call_and_return_conditional_losses_1478344980
	input_108
dense_428_1478344858
dense_428_1478344860
dense_429_1478344885
dense_429_1478344887
dense_430_1478344912
dense_430_1478344914&
"batch_normalization_107_1478344943&
"batch_normalization_107_1478344945&
"batch_normalization_107_1478344947&
"batch_normalization_107_1478344949
dense_431_1478344974
dense_431_1478344976
identity¢/batch_normalization_107/StatefulPartitionedCall¢!dense_428/StatefulPartitionedCall¢!dense_429/StatefulPartitionedCall¢!dense_430/StatefulPartitionedCall¢!dense_431/StatefulPartitionedCallå
flatten_107/PartitionedCallPartitionedCall	input_108*
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
GPU 2J 8 *T
fORM
K__inference_flatten_107_layer_call_and_return_conditional_losses_14783448282
flatten_107/PartitionedCallÄ
!dense_428/StatefulPartitionedCallStatefulPartitionedCall$flatten_107/PartitionedCall:output:0dense_428_1478344858dense_428_1478344860*
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
GPU 2J 8 *R
fMRK
I__inference_dense_428_layer_call_and_return_conditional_losses_14783448472#
!dense_428/StatefulPartitionedCallÊ
!dense_429/StatefulPartitionedCallStatefulPartitionedCall*dense_428/StatefulPartitionedCall:output:0dense_429_1478344885dense_429_1478344887*
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
GPU 2J 8 *R
fMRK
I__inference_dense_429_layer_call_and_return_conditional_losses_14783448742#
!dense_429/StatefulPartitionedCallÊ
!dense_430/StatefulPartitionedCallStatefulPartitionedCall*dense_429/StatefulPartitionedCall:output:0dense_430_1478344912dense_430_1478344914*
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
GPU 2J 8 *R
fMRK
I__inference_dense_430_layer_call_and_return_conditional_losses_14783449012#
!dense_430/StatefulPartitionedCallÚ
/batch_normalization_107/StatefulPartitionedCallStatefulPartitionedCall*dense_430/StatefulPartitionedCall:output:0"batch_normalization_107_1478344943"batch_normalization_107_1478344945"batch_normalization_107_1478344947"batch_normalization_107_1478344949*
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
GPU 2J 8 *`
f[RY
W__inference_batch_normalization_107_layer_call_and_return_conditional_losses_147834477421
/batch_normalization_107/StatefulPartitionedCall×
!dense_431/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_107/StatefulPartitionedCall:output:0dense_431_1478344974dense_431_1478344976*
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
GPU 2J 8 *R
fMRK
I__inference_dense_431_layer_call_and_return_conditional_losses_14783449632#
!dense_431/StatefulPartitionedCallÀ
IdentityIdentity*dense_431/StatefulPartitionedCall:output:00^batch_normalization_107/StatefulPartitionedCall"^dense_428/StatefulPartitionedCall"^dense_429/StatefulPartitionedCall"^dense_430/StatefulPartitionedCall"^dense_431/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2b
/batch_normalization_107/StatefulPartitionedCall/batch_normalization_107/StatefulPartitionedCall2F
!dense_428/StatefulPartitionedCall!dense_428/StatefulPartitionedCall2F
!dense_429/StatefulPartitionedCall!dense_429/StatefulPartitionedCall2F
!dense_430/StatefulPartitionedCall!dense_430/StatefulPartitionedCall2F
!dense_431/StatefulPartitionedCall!dense_431/StatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
#
_user_specified_name	input_108
­§
³
&__inference__traced_restore_1478345794
file_prefix%
!assignvariableop_dense_428_kernel%
!assignvariableop_1_dense_428_bias'
#assignvariableop_2_dense_429_kernel%
!assignvariableop_3_dense_429_bias'
#assignvariableop_4_dense_430_kernel%
!assignvariableop_5_dense_430_bias4
0assignvariableop_6_batch_normalization_107_gamma3
/assignvariableop_7_batch_normalization_107_beta:
6assignvariableop_8_batch_normalization_107_moving_mean>
:assignvariableop_9_batch_normalization_107_moving_variance(
$assignvariableop_10_dense_431_kernel&
"assignvariableop_11_dense_431_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count/
+assignvariableop_19_adam_dense_428_kernel_m-
)assignvariableop_20_adam_dense_428_bias_m/
+assignvariableop_21_adam_dense_429_kernel_m-
)assignvariableop_22_adam_dense_429_bias_m/
+assignvariableop_23_adam_dense_430_kernel_m-
)assignvariableop_24_adam_dense_430_bias_m<
8assignvariableop_25_adam_batch_normalization_107_gamma_m;
7assignvariableop_26_adam_batch_normalization_107_beta_m/
+assignvariableop_27_adam_dense_431_kernel_m-
)assignvariableop_28_adam_dense_431_bias_m/
+assignvariableop_29_adam_dense_428_kernel_v-
)assignvariableop_30_adam_dense_428_bias_v/
+assignvariableop_31_adam_dense_429_kernel_v-
)assignvariableop_32_adam_dense_429_bias_v/
+assignvariableop_33_adam_dense_430_kernel_v-
)assignvariableop_34_adam_dense_430_bias_v<
8assignvariableop_35_adam_batch_normalization_107_gamma_v;
7assignvariableop_36_adam_batch_normalization_107_beta_v/
+assignvariableop_37_adam_dense_431_kernel_v-
)assignvariableop_38_adam_dense_431_bias_v
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

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_dense_428_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_428_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_429_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_429_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_430_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_430_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6µ
AssignVariableOp_6AssignVariableOp0assignvariableop_6_batch_normalization_107_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7´
AssignVariableOp_7AssignVariableOp/assignvariableop_7_batch_normalization_107_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8»
AssignVariableOp_8AssignVariableOp6assignvariableop_8_batch_normalization_107_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¿
AssignVariableOp_9AssignVariableOp:assignvariableop_9_batch_normalization_107_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¬
AssignVariableOp_10AssignVariableOp$assignvariableop_10_dense_431_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ª
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_431_biasIdentity_11:output:0"/device:CPU:0*
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
Identity_19³
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_dense_428_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20±
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_428_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21³
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_429_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22±
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_429_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23³
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_430_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24±
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_430_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25À
AssignVariableOp_25AssignVariableOp8assignvariableop_25_adam_batch_normalization_107_gamma_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¿
AssignVariableOp_26AssignVariableOp7assignvariableop_26_adam_batch_normalization_107_beta_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27³
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_431_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28±
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_431_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29³
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_428_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30±
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_428_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31³
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_429_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32±
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_429_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33³
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_dense_430_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34±
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_430_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35À
AssignVariableOp_35AssignVariableOp8assignvariableop_35_adam_batch_normalization_107_gamma_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¿
AssignVariableOp_36AssignVariableOp7assignvariableop_36_adam_batch_normalization_107_beta_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37³
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_dense_431_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38±
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_dense_431_bias_vIdentity_38:output:0"/device:CPU:0*
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
â
I__inference_dense_430_layer_call_and_return_conditional_losses_1478345416

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
ü	
â
I__inference_dense_428_layer_call_and_return_conditional_losses_1478344847

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
Ý0
Ö
W__inference_batch_normalization_107_layer_call_and_return_conditional_losses_1478344774

inputs
assignmovingavg_1478344749 
assignmovingavg_1_1478344755)
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
moments/Squeeze_1Ð
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg/1478344749*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1478344749*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpö
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg/1478344749*
_output_shapes	
:2
AssignMovingAvg/subí
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg/1478344749*
_output_shapes	
:2
AssignMovingAvg/mul·
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1478344749AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg/1478344749*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÖ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*/
_class%
#!loc:@AssignMovingAvg_1/1478344755*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1478344755*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*/
_class%
#!loc:@AssignMovingAvg_1/1478344755*
_output_shapes	
:2
AssignMovingAvg_1/sub÷
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*/
_class%
#!loc:@AssignMovingAvg_1/1478344755*
_output_shapes	
:2
AssignMovingAvg_1/mulÃ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1478344755AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*/
_class%
#!loc:@AssignMovingAvg_1/1478344755*
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
º	

.__inference_model_107_layer_call_fn_1478345141
	input_108
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
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCall	input_108unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8 *R
fMRK
I__inference_model_107_layer_call_and_return_conditional_losses_14783451142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
#
_user_specified_name	input_108
¯	

.__inference_model_107_layer_call_fn_1478345325

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
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_model_107_layer_call_and_return_conditional_losses_14783450512
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
ü	
â
I__inference_dense_429_layer_call_and_return_conditional_losses_1478344874

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
þ	
â
I__inference_dense_431_layer_call_and_return_conditional_losses_1478344963

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
»
g
K__inference_flatten_107_layer_call_and_return_conditional_losses_1478344828

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
ü	
â
I__inference_dense_428_layer_call_and_return_conditional_losses_1478345376

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
è

W__inference_batch_normalization_107_layer_call_and_return_conditional_losses_1478344807

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
é

.__inference_dense_431_layer_call_fn_1478345527

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
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
GPU 2J 8 *R
fMRK
I__inference_dense_431_layer_call_and_return_conditional_losses_14783449632
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
ë

.__inference_dense_430_layer_call_fn_1478345425

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
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
GPU 2J 8 *R
fMRK
I__inference_dense_430_layer_call_and_return_conditional_losses_14783449012
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
¸	

.__inference_model_107_layer_call_fn_1478345078
	input_108
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
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCall	input_108unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8 *R
fMRK
I__inference_model_107_layer_call_and_return_conditional_losses_14783450512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
#
_user_specified_name	input_108
öp
À

I__inference_model_107_layer_call_and_return_conditional_losses_1478345246

inputs,
(dense_428_matmul_readvariableop_resource-
)dense_428_biasadd_readvariableop_resource,
(dense_429_matmul_readvariableop_resource-
)dense_429_biasadd_readvariableop_resource,
(dense_430_matmul_readvariableop_resource-
)dense_430_biasadd_readvariableop_resource6
2batch_normalization_107_assignmovingavg_14783452148
4batch_normalization_107_assignmovingavg_1_1478345220A
=batch_normalization_107_batchnorm_mul_readvariableop_resource=
9batch_normalization_107_batchnorm_readvariableop_resource,
(dense_431_matmul_readvariableop_resource-
)dense_431_biasadd_readvariableop_resource
identity¢;batch_normalization_107/AssignMovingAvg/AssignSubVariableOp¢6batch_normalization_107/AssignMovingAvg/ReadVariableOp¢=batch_normalization_107/AssignMovingAvg_1/AssignSubVariableOp¢8batch_normalization_107/AssignMovingAvg_1/ReadVariableOp¢0batch_normalization_107/batchnorm/ReadVariableOp¢4batch_normalization_107/batchnorm/mul/ReadVariableOp¢ dense_428/BiasAdd/ReadVariableOp¢dense_428/MatMul/ReadVariableOp¢ dense_429/BiasAdd/ReadVariableOp¢dense_429/MatMul/ReadVariableOp¢ dense_430/BiasAdd/ReadVariableOp¢dense_430/MatMul/ReadVariableOp¢ dense_431/BiasAdd/ReadVariableOp¢dense_431/MatMul/ReadVariableOpw
flatten_107/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_107/Const
flatten_107/ReshapeReshapeinputsflatten_107/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_107/Reshape­
dense_428/MatMul/ReadVariableOpReadVariableOp(dense_428_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_428/MatMul/ReadVariableOp¨
dense_428/MatMulMatMulflatten_107/Reshape:output:0'dense_428/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_428/MatMul«
 dense_428/BiasAdd/ReadVariableOpReadVariableOp)dense_428_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_428/BiasAdd/ReadVariableOpª
dense_428/BiasAddBiasAdddense_428/MatMul:product:0(dense_428/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_428/BiasAddw
dense_428/ReluReludense_428/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_428/Relu­
dense_429/MatMul/ReadVariableOpReadVariableOp(dense_429_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_429/MatMul/ReadVariableOp¨
dense_429/MatMulMatMuldense_428/Relu:activations:0'dense_429/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_429/MatMul«
 dense_429/BiasAdd/ReadVariableOpReadVariableOp)dense_429_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_429/BiasAdd/ReadVariableOpª
dense_429/BiasAddBiasAdddense_429/MatMul:product:0(dense_429/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_429/BiasAddw
dense_429/ReluReludense_429/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_429/Relu­
dense_430/MatMul/ReadVariableOpReadVariableOp(dense_430_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_430/MatMul/ReadVariableOp¨
dense_430/MatMulMatMuldense_429/Relu:activations:0'dense_430/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_430/MatMul«
 dense_430/BiasAdd/ReadVariableOpReadVariableOp)dense_430_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_430/BiasAdd/ReadVariableOpª
dense_430/BiasAddBiasAdddense_430/MatMul:product:0(dense_430/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_430/BiasAddw
dense_430/ReluReludense_430/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_430/Reluº
6batch_normalization_107/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization_107/moments/mean/reduction_indicesî
$batch_normalization_107/moments/meanMeandense_430/Relu:activations:0?batch_normalization_107/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2&
$batch_normalization_107/moments/meanÅ
,batch_normalization_107/moments/StopGradientStopGradient-batch_normalization_107/moments/mean:output:0*
T0*
_output_shapes
:	2.
,batch_normalization_107/moments/StopGradient
1batch_normalization_107/moments/SquaredDifferenceSquaredDifferencedense_430/Relu:activations:05batch_normalization_107/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1batch_normalization_107/moments/SquaredDifferenceÂ
:batch_normalization_107/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2<
:batch_normalization_107/moments/variance/reduction_indices
(batch_normalization_107/moments/varianceMean5batch_normalization_107/moments/SquaredDifference:z:0Cbatch_normalization_107/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2*
(batch_normalization_107/moments/varianceÉ
'batch_normalization_107/moments/SqueezeSqueeze-batch_normalization_107/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2)
'batch_normalization_107/moments/SqueezeÑ
)batch_normalization_107/moments/Squeeze_1Squeeze1batch_normalization_107/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2+
)batch_normalization_107/moments/Squeeze_1
-batch_normalization_107/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*E
_class;
97loc:@batch_normalization_107/AssignMovingAvg/1478345214*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_107/AssignMovingAvg/decayà
6batch_normalization_107/AssignMovingAvg/ReadVariableOpReadVariableOp2batch_normalization_107_assignmovingavg_1478345214*
_output_shapes	
:*
dtype028
6batch_normalization_107/AssignMovingAvg/ReadVariableOpî
+batch_normalization_107/AssignMovingAvg/subSub>batch_normalization_107/AssignMovingAvg/ReadVariableOp:value:00batch_normalization_107/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@batch_normalization_107/AssignMovingAvg/1478345214*
_output_shapes	
:2-
+batch_normalization_107/AssignMovingAvg/subå
+batch_normalization_107/AssignMovingAvg/mulMul/batch_normalization_107/AssignMovingAvg/sub:z:06batch_normalization_107/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@batch_normalization_107/AssignMovingAvg/1478345214*
_output_shapes	
:2-
+batch_normalization_107/AssignMovingAvg/mulÇ
;batch_normalization_107/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp2batch_normalization_107_assignmovingavg_1478345214/batch_normalization_107/AssignMovingAvg/mul:z:07^batch_normalization_107/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*E
_class;
97loc:@batch_normalization_107/AssignMovingAvg/1478345214*
_output_shapes
 *
dtype02=
;batch_normalization_107/AssignMovingAvg/AssignSubVariableOp
/batch_normalization_107/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@batch_normalization_107/AssignMovingAvg_1/1478345220*
_output_shapes
: *
dtype0*
valueB
 *
×#<21
/batch_normalization_107/AssignMovingAvg_1/decayæ
8batch_normalization_107/AssignMovingAvg_1/ReadVariableOpReadVariableOp4batch_normalization_107_assignmovingavg_1_1478345220*
_output_shapes	
:*
dtype02:
8batch_normalization_107/AssignMovingAvg_1/ReadVariableOpø
-batch_normalization_107/AssignMovingAvg_1/subSub@batch_normalization_107/AssignMovingAvg_1/ReadVariableOp:value:02batch_normalization_107/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@batch_normalization_107/AssignMovingAvg_1/1478345220*
_output_shapes	
:2/
-batch_normalization_107/AssignMovingAvg_1/subï
-batch_normalization_107/AssignMovingAvg_1/mulMul1batch_normalization_107/AssignMovingAvg_1/sub:z:08batch_normalization_107/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@batch_normalization_107/AssignMovingAvg_1/1478345220*
_output_shapes	
:2/
-batch_normalization_107/AssignMovingAvg_1/mulÓ
=batch_normalization_107/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp4batch_normalization_107_assignmovingavg_1_14783452201batch_normalization_107/AssignMovingAvg_1/mul:z:09^batch_normalization_107/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@batch_normalization_107/AssignMovingAvg_1/1478345220*
_output_shapes
 *
dtype02?
=batch_normalization_107/AssignMovingAvg_1/AssignSubVariableOp
'batch_normalization_107/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2)
'batch_normalization_107/batchnorm/add/yã
%batch_normalization_107/batchnorm/addAddV22batch_normalization_107/moments/Squeeze_1:output:00batch_normalization_107/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2'
%batch_normalization_107/batchnorm/add¬
'batch_normalization_107/batchnorm/RsqrtRsqrt)batch_normalization_107/batchnorm/add:z:0*
T0*
_output_shapes	
:2)
'batch_normalization_107/batchnorm/Rsqrtç
4batch_normalization_107/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_107_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype026
4batch_normalization_107/batchnorm/mul/ReadVariableOpæ
%batch_normalization_107/batchnorm/mulMul+batch_normalization_107/batchnorm/Rsqrt:y:0<batch_normalization_107/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2'
%batch_normalization_107/batchnorm/mulÕ
'batch_normalization_107/batchnorm/mul_1Muldense_430/Relu:activations:0)batch_normalization_107/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'batch_normalization_107/batchnorm/mul_1Ü
'batch_normalization_107/batchnorm/mul_2Mul0batch_normalization_107/moments/Squeeze:output:0)batch_normalization_107/batchnorm/mul:z:0*
T0*
_output_shapes	
:2)
'batch_normalization_107/batchnorm/mul_2Û
0batch_normalization_107/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_107_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype022
0batch_normalization_107/batchnorm/ReadVariableOpâ
%batch_normalization_107/batchnorm/subSub8batch_normalization_107/batchnorm/ReadVariableOp:value:0+batch_normalization_107/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_107/batchnorm/subæ
'batch_normalization_107/batchnorm/add_1AddV2+batch_normalization_107/batchnorm/mul_1:z:0)batch_normalization_107/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'batch_normalization_107/batchnorm/add_1¬
dense_431/MatMul/ReadVariableOpReadVariableOp(dense_431_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02!
dense_431/MatMul/ReadVariableOp¶
dense_431/MatMulMatMul+batch_normalization_107/batchnorm/add_1:z:0'dense_431/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_431/MatMulª
 dense_431/BiasAdd/ReadVariableOpReadVariableOp)dense_431_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_431/BiasAdd/ReadVariableOp©
dense_431/BiasAddBiasAdddense_431/MatMul:product:0(dense_431/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_431/BiasAdd
dense_431/SoftmaxSoftmaxdense_431/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_431/Softmaxß
IdentityIdentitydense_431/Softmax:softmax:0<^batch_normalization_107/AssignMovingAvg/AssignSubVariableOp7^batch_normalization_107/AssignMovingAvg/ReadVariableOp>^batch_normalization_107/AssignMovingAvg_1/AssignSubVariableOp9^batch_normalization_107/AssignMovingAvg_1/ReadVariableOp1^batch_normalization_107/batchnorm/ReadVariableOp5^batch_normalization_107/batchnorm/mul/ReadVariableOp!^dense_428/BiasAdd/ReadVariableOp ^dense_428/MatMul/ReadVariableOp!^dense_429/BiasAdd/ReadVariableOp ^dense_429/MatMul/ReadVariableOp!^dense_430/BiasAdd/ReadVariableOp ^dense_430/MatMul/ReadVariableOp!^dense_431/BiasAdd/ReadVariableOp ^dense_431/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2z
;batch_normalization_107/AssignMovingAvg/AssignSubVariableOp;batch_normalization_107/AssignMovingAvg/AssignSubVariableOp2p
6batch_normalization_107/AssignMovingAvg/ReadVariableOp6batch_normalization_107/AssignMovingAvg/ReadVariableOp2~
=batch_normalization_107/AssignMovingAvg_1/AssignSubVariableOp=batch_normalization_107/AssignMovingAvg_1/AssignSubVariableOp2t
8batch_normalization_107/AssignMovingAvg_1/ReadVariableOp8batch_normalization_107/AssignMovingAvg_1/ReadVariableOp2d
0batch_normalization_107/batchnorm/ReadVariableOp0batch_normalization_107/batchnorm/ReadVariableOp2l
4batch_normalization_107/batchnorm/mul/ReadVariableOp4batch_normalization_107/batchnorm/mul/ReadVariableOp2D
 dense_428/BiasAdd/ReadVariableOp dense_428/BiasAdd/ReadVariableOp2B
dense_428/MatMul/ReadVariableOpdense_428/MatMul/ReadVariableOp2D
 dense_429/BiasAdd/ReadVariableOp dense_429/BiasAdd/ReadVariableOp2B
dense_429/MatMul/ReadVariableOpdense_429/MatMul/ReadVariableOp2D
 dense_430/BiasAdd/ReadVariableOp dense_430/BiasAdd/ReadVariableOp2B
dense_430/MatMul/ReadVariableOpdense_430/MatMul/ReadVariableOp2D
 dense_431/BiasAdd/ReadVariableOp dense_431/BiasAdd/ReadVariableOp2B
dense_431/MatMul/ReadVariableOpdense_431/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ë

.__inference_dense_428_layer_call_fn_1478345385

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
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
GPU 2J 8 *R
fMRK
I__inference_dense_428_layer_call_and_return_conditional_losses_14783448472
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
Å
¯
<__inference_batch_normalization_107_layer_call_fn_1478345494

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
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_batch_normalization_107_layer_call_and_return_conditional_losses_14783447742
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
U

%__inference__wrapped_model_1478344678
	input_1086
2model_107_dense_428_matmul_readvariableop_resource7
3model_107_dense_428_biasadd_readvariableop_resource6
2model_107_dense_429_matmul_readvariableop_resource7
3model_107_dense_429_biasadd_readvariableop_resource6
2model_107_dense_430_matmul_readvariableop_resource7
3model_107_dense_430_biasadd_readvariableop_resourceG
Cmodel_107_batch_normalization_107_batchnorm_readvariableop_resourceK
Gmodel_107_batch_normalization_107_batchnorm_mul_readvariableop_resourceI
Emodel_107_batch_normalization_107_batchnorm_readvariableop_1_resourceI
Emodel_107_batch_normalization_107_batchnorm_readvariableop_2_resource6
2model_107_dense_431_matmul_readvariableop_resource7
3model_107_dense_431_biasadd_readvariableop_resource
identity¢:model_107/batch_normalization_107/batchnorm/ReadVariableOp¢<model_107/batch_normalization_107/batchnorm/ReadVariableOp_1¢<model_107/batch_normalization_107/batchnorm/ReadVariableOp_2¢>model_107/batch_normalization_107/batchnorm/mul/ReadVariableOp¢*model_107/dense_428/BiasAdd/ReadVariableOp¢)model_107/dense_428/MatMul/ReadVariableOp¢*model_107/dense_429/BiasAdd/ReadVariableOp¢)model_107/dense_429/MatMul/ReadVariableOp¢*model_107/dense_430/BiasAdd/ReadVariableOp¢)model_107/dense_430/MatMul/ReadVariableOp¢*model_107/dense_431/BiasAdd/ReadVariableOp¢)model_107/dense_431/MatMul/ReadVariableOp
model_107/flatten_107/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model_107/flatten_107/Const­
model_107/flatten_107/ReshapeReshape	input_108$model_107/flatten_107/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_107/flatten_107/ReshapeË
)model_107/dense_428/MatMul/ReadVariableOpReadVariableOp2model_107_dense_428_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)model_107/dense_428/MatMul/ReadVariableOpÐ
model_107/dense_428/MatMulMatMul&model_107/flatten_107/Reshape:output:01model_107/dense_428/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_107/dense_428/MatMulÉ
*model_107/dense_428/BiasAdd/ReadVariableOpReadVariableOp3model_107_dense_428_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*model_107/dense_428/BiasAdd/ReadVariableOpÒ
model_107/dense_428/BiasAddBiasAdd$model_107/dense_428/MatMul:product:02model_107/dense_428/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_107/dense_428/BiasAdd
model_107/dense_428/ReluRelu$model_107/dense_428/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_107/dense_428/ReluË
)model_107/dense_429/MatMul/ReadVariableOpReadVariableOp2model_107_dense_429_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)model_107/dense_429/MatMul/ReadVariableOpÐ
model_107/dense_429/MatMulMatMul&model_107/dense_428/Relu:activations:01model_107/dense_429/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_107/dense_429/MatMulÉ
*model_107/dense_429/BiasAdd/ReadVariableOpReadVariableOp3model_107_dense_429_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*model_107/dense_429/BiasAdd/ReadVariableOpÒ
model_107/dense_429/BiasAddBiasAdd$model_107/dense_429/MatMul:product:02model_107/dense_429/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_107/dense_429/BiasAdd
model_107/dense_429/ReluRelu$model_107/dense_429/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_107/dense_429/ReluË
)model_107/dense_430/MatMul/ReadVariableOpReadVariableOp2model_107_dense_430_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)model_107/dense_430/MatMul/ReadVariableOpÐ
model_107/dense_430/MatMulMatMul&model_107/dense_429/Relu:activations:01model_107/dense_430/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_107/dense_430/MatMulÉ
*model_107/dense_430/BiasAdd/ReadVariableOpReadVariableOp3model_107_dense_430_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*model_107/dense_430/BiasAdd/ReadVariableOpÒ
model_107/dense_430/BiasAddBiasAdd$model_107/dense_430/MatMul:product:02model_107/dense_430/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_107/dense_430/BiasAdd
model_107/dense_430/ReluRelu$model_107/dense_430/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_107/dense_430/Reluù
:model_107/batch_normalization_107/batchnorm/ReadVariableOpReadVariableOpCmodel_107_batch_normalization_107_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02<
:model_107/batch_normalization_107/batchnorm/ReadVariableOp«
1model_107/batch_normalization_107/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:23
1model_107/batch_normalization_107/batchnorm/add/y
/model_107/batch_normalization_107/batchnorm/addAddV2Bmodel_107/batch_normalization_107/batchnorm/ReadVariableOp:value:0:model_107/batch_normalization_107/batchnorm/add/y:output:0*
T0*
_output_shapes	
:21
/model_107/batch_normalization_107/batchnorm/addÊ
1model_107/batch_normalization_107/batchnorm/RsqrtRsqrt3model_107/batch_normalization_107/batchnorm/add:z:0*
T0*
_output_shapes	
:23
1model_107/batch_normalization_107/batchnorm/Rsqrt
>model_107/batch_normalization_107/batchnorm/mul/ReadVariableOpReadVariableOpGmodel_107_batch_normalization_107_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02@
>model_107/batch_normalization_107/batchnorm/mul/ReadVariableOp
/model_107/batch_normalization_107/batchnorm/mulMul5model_107/batch_normalization_107/batchnorm/Rsqrt:y:0Fmodel_107/batch_normalization_107/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:21
/model_107/batch_normalization_107/batchnorm/mulý
1model_107/batch_normalization_107/batchnorm/mul_1Mul&model_107/dense_430/Relu:activations:03model_107/batch_normalization_107/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1model_107/batch_normalization_107/batchnorm/mul_1ÿ
<model_107/batch_normalization_107/batchnorm/ReadVariableOp_1ReadVariableOpEmodel_107_batch_normalization_107_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02>
<model_107/batch_normalization_107/batchnorm/ReadVariableOp_1
1model_107/batch_normalization_107/batchnorm/mul_2MulDmodel_107/batch_normalization_107/batchnorm/ReadVariableOp_1:value:03model_107/batch_normalization_107/batchnorm/mul:z:0*
T0*
_output_shapes	
:23
1model_107/batch_normalization_107/batchnorm/mul_2ÿ
<model_107/batch_normalization_107/batchnorm/ReadVariableOp_2ReadVariableOpEmodel_107_batch_normalization_107_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02>
<model_107/batch_normalization_107/batchnorm/ReadVariableOp_2
/model_107/batch_normalization_107/batchnorm/subSubDmodel_107/batch_normalization_107/batchnorm/ReadVariableOp_2:value:05model_107/batch_normalization_107/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:21
/model_107/batch_normalization_107/batchnorm/sub
1model_107/batch_normalization_107/batchnorm/add_1AddV25model_107/batch_normalization_107/batchnorm/mul_1:z:03model_107/batch_normalization_107/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1model_107/batch_normalization_107/batchnorm/add_1Ê
)model_107/dense_431/MatMul/ReadVariableOpReadVariableOp2model_107_dense_431_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)model_107/dense_431/MatMul/ReadVariableOpÞ
model_107/dense_431/MatMulMatMul5model_107/batch_normalization_107/batchnorm/add_1:z:01model_107/dense_431/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_107/dense_431/MatMulÈ
*model_107/dense_431/BiasAdd/ReadVariableOpReadVariableOp3model_107_dense_431_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*model_107/dense_431/BiasAdd/ReadVariableOpÑ
model_107/dense_431/BiasAddBiasAdd$model_107/dense_431/MatMul:product:02model_107/dense_431/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_107/dense_431/BiasAdd
model_107/dense_431/SoftmaxSoftmax$model_107/dense_431/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_107/dense_431/SoftmaxÙ
IdentityIdentity%model_107/dense_431/Softmax:softmax:0;^model_107/batch_normalization_107/batchnorm/ReadVariableOp=^model_107/batch_normalization_107/batchnorm/ReadVariableOp_1=^model_107/batch_normalization_107/batchnorm/ReadVariableOp_2?^model_107/batch_normalization_107/batchnorm/mul/ReadVariableOp+^model_107/dense_428/BiasAdd/ReadVariableOp*^model_107/dense_428/MatMul/ReadVariableOp+^model_107/dense_429/BiasAdd/ReadVariableOp*^model_107/dense_429/MatMul/ReadVariableOp+^model_107/dense_430/BiasAdd/ReadVariableOp*^model_107/dense_430/MatMul/ReadVariableOp+^model_107/dense_431/BiasAdd/ReadVariableOp*^model_107/dense_431/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2x
:model_107/batch_normalization_107/batchnorm/ReadVariableOp:model_107/batch_normalization_107/batchnorm/ReadVariableOp2|
<model_107/batch_normalization_107/batchnorm/ReadVariableOp_1<model_107/batch_normalization_107/batchnorm/ReadVariableOp_12|
<model_107/batch_normalization_107/batchnorm/ReadVariableOp_2<model_107/batch_normalization_107/batchnorm/ReadVariableOp_22
>model_107/batch_normalization_107/batchnorm/mul/ReadVariableOp>model_107/batch_normalization_107/batchnorm/mul/ReadVariableOp2X
*model_107/dense_428/BiasAdd/ReadVariableOp*model_107/dense_428/BiasAdd/ReadVariableOp2V
)model_107/dense_428/MatMul/ReadVariableOp)model_107/dense_428/MatMul/ReadVariableOp2X
*model_107/dense_429/BiasAdd/ReadVariableOp*model_107/dense_429/BiasAdd/ReadVariableOp2V
)model_107/dense_429/MatMul/ReadVariableOp)model_107/dense_429/MatMul/ReadVariableOp2X
*model_107/dense_430/BiasAdd/ReadVariableOp*model_107/dense_430/BiasAdd/ReadVariableOp2V
)model_107/dense_430/MatMul/ReadVariableOp)model_107/dense_430/MatMul/ReadVariableOp2X
*model_107/dense_431/BiasAdd/ReadVariableOp*model_107/dense_431/BiasAdd/ReadVariableOp2V
)model_107/dense_431/MatMul/ReadVariableOp)model_107/dense_431/MatMul/ReadVariableOp:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
#
_user_specified_name	input_108
¼"

I__inference_model_107_layer_call_and_return_conditional_losses_1478345014
	input_108
dense_428_1478344984
dense_428_1478344986
dense_429_1478344989
dense_429_1478344991
dense_430_1478344994
dense_430_1478344996&
"batch_normalization_107_1478344999&
"batch_normalization_107_1478345001&
"batch_normalization_107_1478345003&
"batch_normalization_107_1478345005
dense_431_1478345008
dense_431_1478345010
identity¢/batch_normalization_107/StatefulPartitionedCall¢!dense_428/StatefulPartitionedCall¢!dense_429/StatefulPartitionedCall¢!dense_430/StatefulPartitionedCall¢!dense_431/StatefulPartitionedCallå
flatten_107/PartitionedCallPartitionedCall	input_108*
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
GPU 2J 8 *T
fORM
K__inference_flatten_107_layer_call_and_return_conditional_losses_14783448282
flatten_107/PartitionedCallÄ
!dense_428/StatefulPartitionedCallStatefulPartitionedCall$flatten_107/PartitionedCall:output:0dense_428_1478344984dense_428_1478344986*
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
GPU 2J 8 *R
fMRK
I__inference_dense_428_layer_call_and_return_conditional_losses_14783448472#
!dense_428/StatefulPartitionedCallÊ
!dense_429/StatefulPartitionedCallStatefulPartitionedCall*dense_428/StatefulPartitionedCall:output:0dense_429_1478344989dense_429_1478344991*
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
GPU 2J 8 *R
fMRK
I__inference_dense_429_layer_call_and_return_conditional_losses_14783448742#
!dense_429/StatefulPartitionedCallÊ
!dense_430/StatefulPartitionedCallStatefulPartitionedCall*dense_429/StatefulPartitionedCall:output:0dense_430_1478344994dense_430_1478344996*
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
GPU 2J 8 *R
fMRK
I__inference_dense_430_layer_call_and_return_conditional_losses_14783449012#
!dense_430/StatefulPartitionedCallÜ
/batch_normalization_107/StatefulPartitionedCallStatefulPartitionedCall*dense_430/StatefulPartitionedCall:output:0"batch_normalization_107_1478344999"batch_normalization_107_1478345001"batch_normalization_107_1478345003"batch_normalization_107_1478345005*
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
GPU 2J 8 *`
f[RY
W__inference_batch_normalization_107_layer_call_and_return_conditional_losses_147834480721
/batch_normalization_107/StatefulPartitionedCall×
!dense_431/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_107/StatefulPartitionedCall:output:0dense_431_1478345008dense_431_1478345010*
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
GPU 2J 8 *R
fMRK
I__inference_dense_431_layer_call_and_return_conditional_losses_14783449632#
!dense_431/StatefulPartitionedCallÀ
IdentityIdentity*dense_431/StatefulPartitionedCall:output:00^batch_normalization_107/StatefulPartitionedCall"^dense_428/StatefulPartitionedCall"^dense_429/StatefulPartitionedCall"^dense_430/StatefulPartitionedCall"^dense_431/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2b
/batch_normalization_107/StatefulPartitionedCall/batch_normalization_107/StatefulPartitionedCall2F
!dense_428/StatefulPartitionedCall!dense_428/StatefulPartitionedCall2F
!dense_429/StatefulPartitionedCall!dense_429/StatefulPartitionedCall2F
!dense_430/StatefulPartitionedCall!dense_430/StatefulPartitionedCall2F
!dense_431/StatefulPartitionedCall!dense_431/StatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
#
_user_specified_name	input_108
±"

I__inference_model_107_layer_call_and_return_conditional_losses_1478345051

inputs
dense_428_1478345021
dense_428_1478345023
dense_429_1478345026
dense_429_1478345028
dense_430_1478345031
dense_430_1478345033&
"batch_normalization_107_1478345036&
"batch_normalization_107_1478345038&
"batch_normalization_107_1478345040&
"batch_normalization_107_1478345042
dense_431_1478345045
dense_431_1478345047
identity¢/batch_normalization_107/StatefulPartitionedCall¢!dense_428/StatefulPartitionedCall¢!dense_429/StatefulPartitionedCall¢!dense_430/StatefulPartitionedCall¢!dense_431/StatefulPartitionedCallâ
flatten_107/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8 *T
fORM
K__inference_flatten_107_layer_call_and_return_conditional_losses_14783448282
flatten_107/PartitionedCallÄ
!dense_428/StatefulPartitionedCallStatefulPartitionedCall$flatten_107/PartitionedCall:output:0dense_428_1478345021dense_428_1478345023*
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
GPU 2J 8 *R
fMRK
I__inference_dense_428_layer_call_and_return_conditional_losses_14783448472#
!dense_428/StatefulPartitionedCallÊ
!dense_429/StatefulPartitionedCallStatefulPartitionedCall*dense_428/StatefulPartitionedCall:output:0dense_429_1478345026dense_429_1478345028*
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
GPU 2J 8 *R
fMRK
I__inference_dense_429_layer_call_and_return_conditional_losses_14783448742#
!dense_429/StatefulPartitionedCallÊ
!dense_430/StatefulPartitionedCallStatefulPartitionedCall*dense_429/StatefulPartitionedCall:output:0dense_430_1478345031dense_430_1478345033*
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
GPU 2J 8 *R
fMRK
I__inference_dense_430_layer_call_and_return_conditional_losses_14783449012#
!dense_430/StatefulPartitionedCallÚ
/batch_normalization_107/StatefulPartitionedCallStatefulPartitionedCall*dense_430/StatefulPartitionedCall:output:0"batch_normalization_107_1478345036"batch_normalization_107_1478345038"batch_normalization_107_1478345040"batch_normalization_107_1478345042*
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
GPU 2J 8 *`
f[RY
W__inference_batch_normalization_107_layer_call_and_return_conditional_losses_147834477421
/batch_normalization_107/StatefulPartitionedCall×
!dense_431/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_107/StatefulPartitionedCall:output:0dense_431_1478345045dense_431_1478345047*
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
GPU 2J 8 *R
fMRK
I__inference_dense_431_layer_call_and_return_conditional_losses_14783449632#
!dense_431/StatefulPartitionedCallÀ
IdentityIdentity*dense_431/StatefulPartitionedCall:output:00^batch_normalization_107/StatefulPartitionedCall"^dense_428/StatefulPartitionedCall"^dense_429/StatefulPartitionedCall"^dense_430/StatefulPartitionedCall"^dense_431/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2b
/batch_normalization_107/StatefulPartitionedCall/batch_normalization_107/StatefulPartitionedCall2F
!dense_428/StatefulPartitionedCall!dense_428/StatefulPartitionedCall2F
!dense_429/StatefulPartitionedCall!dense_429/StatefulPartitionedCall2F
!dense_430/StatefulPartitionedCall!dense_430/StatefulPartitionedCall2F
!dense_431/StatefulPartitionedCall!dense_431/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
»
g
K__inference_flatten_107_layer_call_and_return_conditional_losses_1478345360

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
ÛG
È	
I__inference_model_107_layer_call_and_return_conditional_losses_1478345296

inputs,
(dense_428_matmul_readvariableop_resource-
)dense_428_biasadd_readvariableop_resource,
(dense_429_matmul_readvariableop_resource-
)dense_429_biasadd_readvariableop_resource,
(dense_430_matmul_readvariableop_resource-
)dense_430_biasadd_readvariableop_resource=
9batch_normalization_107_batchnorm_readvariableop_resourceA
=batch_normalization_107_batchnorm_mul_readvariableop_resource?
;batch_normalization_107_batchnorm_readvariableop_1_resource?
;batch_normalization_107_batchnorm_readvariableop_2_resource,
(dense_431_matmul_readvariableop_resource-
)dense_431_biasadd_readvariableop_resource
identity¢0batch_normalization_107/batchnorm/ReadVariableOp¢2batch_normalization_107/batchnorm/ReadVariableOp_1¢2batch_normalization_107/batchnorm/ReadVariableOp_2¢4batch_normalization_107/batchnorm/mul/ReadVariableOp¢ dense_428/BiasAdd/ReadVariableOp¢dense_428/MatMul/ReadVariableOp¢ dense_429/BiasAdd/ReadVariableOp¢dense_429/MatMul/ReadVariableOp¢ dense_430/BiasAdd/ReadVariableOp¢dense_430/MatMul/ReadVariableOp¢ dense_431/BiasAdd/ReadVariableOp¢dense_431/MatMul/ReadVariableOpw
flatten_107/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_107/Const
flatten_107/ReshapeReshapeinputsflatten_107/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_107/Reshape­
dense_428/MatMul/ReadVariableOpReadVariableOp(dense_428_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_428/MatMul/ReadVariableOp¨
dense_428/MatMulMatMulflatten_107/Reshape:output:0'dense_428/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_428/MatMul«
 dense_428/BiasAdd/ReadVariableOpReadVariableOp)dense_428_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_428/BiasAdd/ReadVariableOpª
dense_428/BiasAddBiasAdddense_428/MatMul:product:0(dense_428/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_428/BiasAddw
dense_428/ReluReludense_428/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_428/Relu­
dense_429/MatMul/ReadVariableOpReadVariableOp(dense_429_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_429/MatMul/ReadVariableOp¨
dense_429/MatMulMatMuldense_428/Relu:activations:0'dense_429/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_429/MatMul«
 dense_429/BiasAdd/ReadVariableOpReadVariableOp)dense_429_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_429/BiasAdd/ReadVariableOpª
dense_429/BiasAddBiasAdddense_429/MatMul:product:0(dense_429/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_429/BiasAddw
dense_429/ReluReludense_429/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_429/Relu­
dense_430/MatMul/ReadVariableOpReadVariableOp(dense_430_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_430/MatMul/ReadVariableOp¨
dense_430/MatMulMatMuldense_429/Relu:activations:0'dense_430/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_430/MatMul«
 dense_430/BiasAdd/ReadVariableOpReadVariableOp)dense_430_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_430/BiasAdd/ReadVariableOpª
dense_430/BiasAddBiasAdddense_430/MatMul:product:0(dense_430/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_430/BiasAddw
dense_430/ReluReludense_430/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_430/ReluÛ
0batch_normalization_107/batchnorm/ReadVariableOpReadVariableOp9batch_normalization_107_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype022
0batch_normalization_107/batchnorm/ReadVariableOp
'batch_normalization_107/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2)
'batch_normalization_107/batchnorm/add/yé
%batch_normalization_107/batchnorm/addAddV28batch_normalization_107/batchnorm/ReadVariableOp:value:00batch_normalization_107/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2'
%batch_normalization_107/batchnorm/add¬
'batch_normalization_107/batchnorm/RsqrtRsqrt)batch_normalization_107/batchnorm/add:z:0*
T0*
_output_shapes	
:2)
'batch_normalization_107/batchnorm/Rsqrtç
4batch_normalization_107/batchnorm/mul/ReadVariableOpReadVariableOp=batch_normalization_107_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype026
4batch_normalization_107/batchnorm/mul/ReadVariableOpæ
%batch_normalization_107/batchnorm/mulMul+batch_normalization_107/batchnorm/Rsqrt:y:0<batch_normalization_107/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2'
%batch_normalization_107/batchnorm/mulÕ
'batch_normalization_107/batchnorm/mul_1Muldense_430/Relu:activations:0)batch_normalization_107/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'batch_normalization_107/batchnorm/mul_1á
2batch_normalization_107/batchnorm/ReadVariableOp_1ReadVariableOp;batch_normalization_107_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype024
2batch_normalization_107/batchnorm/ReadVariableOp_1æ
'batch_normalization_107/batchnorm/mul_2Mul:batch_normalization_107/batchnorm/ReadVariableOp_1:value:0)batch_normalization_107/batchnorm/mul:z:0*
T0*
_output_shapes	
:2)
'batch_normalization_107/batchnorm/mul_2á
2batch_normalization_107/batchnorm/ReadVariableOp_2ReadVariableOp;batch_normalization_107_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype024
2batch_normalization_107/batchnorm/ReadVariableOp_2ä
%batch_normalization_107/batchnorm/subSub:batch_normalization_107/batchnorm/ReadVariableOp_2:value:0+batch_normalization_107/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2'
%batch_normalization_107/batchnorm/subæ
'batch_normalization_107/batchnorm/add_1AddV2+batch_normalization_107/batchnorm/mul_1:z:0)batch_normalization_107/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'batch_normalization_107/batchnorm/add_1¬
dense_431/MatMul/ReadVariableOpReadVariableOp(dense_431_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02!
dense_431/MatMul/ReadVariableOp¶
dense_431/MatMulMatMul+batch_normalization_107/batchnorm/add_1:z:0'dense_431/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_431/MatMulª
 dense_431/BiasAdd/ReadVariableOpReadVariableOp)dense_431_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_431/BiasAdd/ReadVariableOp©
dense_431/BiasAddBiasAdddense_431/MatMul:product:0(dense_431/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_431/BiasAdd
dense_431/SoftmaxSoftmaxdense_431/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_431/Softmax×
IdentityIdentitydense_431/Softmax:softmax:01^batch_normalization_107/batchnorm/ReadVariableOp3^batch_normalization_107/batchnorm/ReadVariableOp_13^batch_normalization_107/batchnorm/ReadVariableOp_25^batch_normalization_107/batchnorm/mul/ReadVariableOp!^dense_428/BiasAdd/ReadVariableOp ^dense_428/MatMul/ReadVariableOp!^dense_429/BiasAdd/ReadVariableOp ^dense_429/MatMul/ReadVariableOp!^dense_430/BiasAdd/ReadVariableOp ^dense_430/MatMul/ReadVariableOp!^dense_431/BiasAdd/ReadVariableOp ^dense_431/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2d
0batch_normalization_107/batchnorm/ReadVariableOp0batch_normalization_107/batchnorm/ReadVariableOp2h
2batch_normalization_107/batchnorm/ReadVariableOp_12batch_normalization_107/batchnorm/ReadVariableOp_12h
2batch_normalization_107/batchnorm/ReadVariableOp_22batch_normalization_107/batchnorm/ReadVariableOp_22l
4batch_normalization_107/batchnorm/mul/ReadVariableOp4batch_normalization_107/batchnorm/mul/ReadVariableOp2D
 dense_428/BiasAdd/ReadVariableOp dense_428/BiasAdd/ReadVariableOp2B
dense_428/MatMul/ReadVariableOpdense_428/MatMul/ReadVariableOp2D
 dense_429/BiasAdd/ReadVariableOp dense_429/BiasAdd/ReadVariableOp2B
dense_429/MatMul/ReadVariableOpdense_429/MatMul/ReadVariableOp2D
 dense_430/BiasAdd/ReadVariableOp dense_430/BiasAdd/ReadVariableOp2B
dense_430/MatMul/ReadVariableOpdense_430/MatMul/ReadVariableOp2D
 dense_431/BiasAdd/ReadVariableOp dense_431/BiasAdd/ReadVariableOp2B
dense_431/MatMul/ReadVariableOpdense_431/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
ë

.__inference_dense_429_layer_call_fn_1478345405

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
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
GPU 2J 8 *R
fMRK
I__inference_dense_429_layer_call_and_return_conditional_losses_14783448742
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
ª
L
0__inference_flatten_107_layer_call_fn_1478345365

inputs
identityÊ
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
GPU 2J 8 *T
fORM
K__inference_flatten_107_layer_call_and_return_conditional_losses_14783448282
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
è

W__inference_batch_normalization_107_layer_call_and_return_conditional_losses_1478345481

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
	

(__inference_signature_wrapper_1478345180
	input_108
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
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCall	input_108unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8 *.
f)R'
%__inference__wrapped_model_14783446782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
#
_user_specified_name	input_108
³"

I__inference_model_107_layer_call_and_return_conditional_losses_1478345114

inputs
dense_428_1478345084
dense_428_1478345086
dense_429_1478345089
dense_429_1478345091
dense_430_1478345094
dense_430_1478345096&
"batch_normalization_107_1478345099&
"batch_normalization_107_1478345101&
"batch_normalization_107_1478345103&
"batch_normalization_107_1478345105
dense_431_1478345108
dense_431_1478345110
identity¢/batch_normalization_107/StatefulPartitionedCall¢!dense_428/StatefulPartitionedCall¢!dense_429/StatefulPartitionedCall¢!dense_430/StatefulPartitionedCall¢!dense_431/StatefulPartitionedCallâ
flatten_107/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8 *T
fORM
K__inference_flatten_107_layer_call_and_return_conditional_losses_14783448282
flatten_107/PartitionedCallÄ
!dense_428/StatefulPartitionedCallStatefulPartitionedCall$flatten_107/PartitionedCall:output:0dense_428_1478345084dense_428_1478345086*
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
GPU 2J 8 *R
fMRK
I__inference_dense_428_layer_call_and_return_conditional_losses_14783448472#
!dense_428/StatefulPartitionedCallÊ
!dense_429/StatefulPartitionedCallStatefulPartitionedCall*dense_428/StatefulPartitionedCall:output:0dense_429_1478345089dense_429_1478345091*
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
GPU 2J 8 *R
fMRK
I__inference_dense_429_layer_call_and_return_conditional_losses_14783448742#
!dense_429/StatefulPartitionedCallÊ
!dense_430/StatefulPartitionedCallStatefulPartitionedCall*dense_429/StatefulPartitionedCall:output:0dense_430_1478345094dense_430_1478345096*
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
GPU 2J 8 *R
fMRK
I__inference_dense_430_layer_call_and_return_conditional_losses_14783449012#
!dense_430/StatefulPartitionedCallÜ
/batch_normalization_107/StatefulPartitionedCallStatefulPartitionedCall*dense_430/StatefulPartitionedCall:output:0"batch_normalization_107_1478345099"batch_normalization_107_1478345101"batch_normalization_107_1478345103"batch_normalization_107_1478345105*
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
GPU 2J 8 *`
f[RY
W__inference_batch_normalization_107_layer_call_and_return_conditional_losses_147834480721
/batch_normalization_107/StatefulPartitionedCall×
!dense_431/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_107/StatefulPartitionedCall:output:0dense_431_1478345108dense_431_1478345110*
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
GPU 2J 8 *R
fMRK
I__inference_dense_431_layer_call_and_return_conditional_losses_14783449632#
!dense_431/StatefulPartitionedCallÀ
IdentityIdentity*dense_431/StatefulPartitionedCall:output:00^batch_normalization_107/StatefulPartitionedCall"^dense_428/StatefulPartitionedCall"^dense_429/StatefulPartitionedCall"^dense_430/StatefulPartitionedCall"^dense_431/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2b
/batch_normalization_107/StatefulPartitionedCall/batch_normalization_107/StatefulPartitionedCall2F
!dense_428/StatefulPartitionedCall!dense_428/StatefulPartitionedCall2F
!dense_429/StatefulPartitionedCall!dense_429/StatefulPartitionedCall2F
!dense_430/StatefulPartitionedCall!dense_430/StatefulPartitionedCall2F
!dense_431/StatefulPartitionedCall!dense_431/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
±	

.__inference_model_107_layer_call_fn_1478345354

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
identity¢StatefulPartitionedCallü
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
GPU 2J 8 *R
fMRK
I__inference_model_107_layer_call_and_return_conditional_losses_14783451142
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
Ý0
Ö
W__inference_batch_normalization_107_layer_call_and_return_conditional_losses_1478345461

inputs
assignmovingavg_1478345436 
assignmovingavg_1_1478345442)
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
moments/Squeeze_1Ð
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg/1478345436*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1478345436*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpö
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg/1478345436*
_output_shapes	
:2
AssignMovingAvg/subí
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg/1478345436*
_output_shapes	
:2
AssignMovingAvg/mul·
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1478345436AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg/1478345436*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÖ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*/
_class%
#!loc:@AssignMovingAvg_1/1478345442*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1478345442*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*/
_class%
#!loc:@AssignMovingAvg_1/1478345442*
_output_shapes	
:2
AssignMovingAvg_1/sub÷
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*/
_class%
#!loc:@AssignMovingAvg_1/1478345442*
_output_shapes	
:2
AssignMovingAvg_1/mulÃ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1478345442AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*/
_class%
#!loc:@AssignMovingAvg_1/1478345442*
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
U
¥
#__inference__traced_save_1478345667
file_prefix/
+savev2_dense_428_kernel_read_readvariableop-
)savev2_dense_428_bias_read_readvariableop/
+savev2_dense_429_kernel_read_readvariableop-
)savev2_dense_429_bias_read_readvariableop/
+savev2_dense_430_kernel_read_readvariableop-
)savev2_dense_430_bias_read_readvariableop<
8savev2_batch_normalization_107_gamma_read_readvariableop;
7savev2_batch_normalization_107_beta_read_readvariableopB
>savev2_batch_normalization_107_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_107_moving_variance_read_readvariableop/
+savev2_dense_431_kernel_read_readvariableop-
)savev2_dense_431_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_428_kernel_m_read_readvariableop4
0savev2_adam_dense_428_bias_m_read_readvariableop6
2savev2_adam_dense_429_kernel_m_read_readvariableop4
0savev2_adam_dense_429_bias_m_read_readvariableop6
2savev2_adam_dense_430_kernel_m_read_readvariableop4
0savev2_adam_dense_430_bias_m_read_readvariableopC
?savev2_adam_batch_normalization_107_gamma_m_read_readvariableopB
>savev2_adam_batch_normalization_107_beta_m_read_readvariableop6
2savev2_adam_dense_431_kernel_m_read_readvariableop4
0savev2_adam_dense_431_bias_m_read_readvariableop6
2savev2_adam_dense_428_kernel_v_read_readvariableop4
0savev2_adam_dense_428_bias_v_read_readvariableop6
2savev2_adam_dense_429_kernel_v_read_readvariableop4
0savev2_adam_dense_429_bias_v_read_readvariableop6
2savev2_adam_dense_430_kernel_v_read_readvariableop4
0savev2_adam_dense_430_bias_v_read_readvariableopC
?savev2_adam_batch_normalization_107_gamma_v_read_readvariableopB
>savev2_adam_batch_normalization_107_beta_v_read_readvariableop6
2savev2_adam_dense_431_kernel_v_read_readvariableop4
0savev2_adam_dense_431_bias_v_read_readvariableop
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
SaveV2/shape_and_slicesú
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_428_kernel_read_readvariableop)savev2_dense_428_bias_read_readvariableop+savev2_dense_429_kernel_read_readvariableop)savev2_dense_429_bias_read_readvariableop+savev2_dense_430_kernel_read_readvariableop)savev2_dense_430_bias_read_readvariableop8savev2_batch_normalization_107_gamma_read_readvariableop7savev2_batch_normalization_107_beta_read_readvariableop>savev2_batch_normalization_107_moving_mean_read_readvariableopBsavev2_batch_normalization_107_moving_variance_read_readvariableop+savev2_dense_431_kernel_read_readvariableop)savev2_dense_431_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_428_kernel_m_read_readvariableop0savev2_adam_dense_428_bias_m_read_readvariableop2savev2_adam_dense_429_kernel_m_read_readvariableop0savev2_adam_dense_429_bias_m_read_readvariableop2savev2_adam_dense_430_kernel_m_read_readvariableop0savev2_adam_dense_430_bias_m_read_readvariableop?savev2_adam_batch_normalization_107_gamma_m_read_readvariableop>savev2_adam_batch_normalization_107_beta_m_read_readvariableop2savev2_adam_dense_431_kernel_m_read_readvariableop0savev2_adam_dense_431_bias_m_read_readvariableop2savev2_adam_dense_428_kernel_v_read_readvariableop0savev2_adam_dense_428_bias_v_read_readvariableop2savev2_adam_dense_429_kernel_v_read_readvariableop0savev2_adam_dense_429_bias_v_read_readvariableop2savev2_adam_dense_430_kernel_v_read_readvariableop0savev2_adam_dense_430_bias_v_read_readvariableop?savev2_adam_batch_normalization_107_gamma_v_read_readvariableop>savev2_adam_batch_normalization_107_beta_v_read_readvariableop2savev2_adam_dense_431_kernel_v_read_readvariableop0savev2_adam_dense_431_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
Ç
¯
<__inference_batch_normalization_107_layer_call_fn_1478345507

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¢
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
GPU 2J 8 *`
f[RY
W__inference_batch_normalization_107_layer_call_and_return_conditional_losses_14783448072
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
ü	
â
I__inference_dense_430_layer_call_and_return_conditional_losses_1478344901

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
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*´
serving_default 
C
	input_1086
serving_default_input_108:0ÿÿÿÿÿÿÿÿÿ4=
	dense_4310
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¹ä
>
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
v__call__"ê:
_tf_keras_networkÎ:{"class_name": "Functional", "name": "model_107", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_107", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_108"}, "name": "input_108", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_107", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_107", "inbound_nodes": [[["input_108", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_428", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_428", "inbound_nodes": [[["flatten_107", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_429", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_429", "inbound_nodes": [[["dense_428", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_430", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_430", "inbound_nodes": [[["dense_429", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_107", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_107", "inbound_nodes": [[["dense_430", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_431", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_431", "inbound_nodes": [[["batch_normalization_107", 0, 0, {}]]]}], "input_layers": [["input_108", 0, 0]], "output_layers": [["dense_431", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 52, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_107", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_108"}, "name": "input_108", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_107", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_107", "inbound_nodes": [[["input_108", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_428", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_428", "inbound_nodes": [[["flatten_107", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_429", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_429", "inbound_nodes": [[["dense_428", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_430", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_430", "inbound_nodes": [[["dense_429", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_107", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_107", "inbound_nodes": [[["dense_430", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_431", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_431", "inbound_nodes": [[["batch_normalization_107", 0, 0, {}]]]}], "input_layers": [["input_108", 0, 0]], "output_layers": [["dense_431", 0, 0]]}}, "training_config": {"loss": "loss_func", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "clipnorm": 1, "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
õ"ò
_tf_keras_input_layerÒ{"class_name": "InputLayer", "name": "input_108", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_108"}}
ê
regularization_losses
	variables
trainable_variables
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"Û
_tf_keras_layerÁ{"class_name": "Flatten", "name": "flatten_107", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_107", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*y&call_and_return_all_conditional_losses
z__call__"ó
_tf_keras_layerÙ{"class_name": "Dense", "name": "dense_428", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_428", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 156}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 156]}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*{&call_and_return_all_conditional_losses
|__call__"ó
_tf_keras_layerÙ{"class_name": "Dense", "name": "dense_429", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_429", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}


kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
*}&call_and_return_all_conditional_losses
~__call__"ó
_tf_keras_layerÙ{"class_name": "Dense", "name": "dense_430", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_430", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
¹	
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
__call__"ä
_tf_keras_layerÊ{"class_name": "BatchNormalization", "name": "batch_normalization_107", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_107", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ú

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+&call_and_return_all_conditional_losses
__call__"Ó
_tf_keras_layer¹{"class_name": "Dense", "name": "dense_431", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_431", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
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
$:"
2dense_428/kernel
:2dense_428/bias
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
$:"
2dense_429/kernel
:2dense_429/bias
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
$:"
2dense_430/kernel
:2dense_430/bias
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
,:*2batch_normalization_107/gamma
+:)2batch_normalization_107/beta
4:2 (2#batch_normalization_107/moving_mean
8:6 (2'batch_normalization_107/moving_variance
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
#:!	2dense_431/kernel
:2dense_431/bias
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
):'
2Adam/dense_428/kernel/m
": 2Adam/dense_428/bias/m
):'
2Adam/dense_429/kernel/m
": 2Adam/dense_429/bias/m
):'
2Adam/dense_430/kernel/m
": 2Adam/dense_430/bias/m
1:/2$Adam/batch_normalization_107/gamma/m
0:.2#Adam/batch_normalization_107/beta/m
(:&	2Adam/dense_431/kernel/m
!:2Adam/dense_431/bias/m
):'
2Adam/dense_428/kernel/v
": 2Adam/dense_428/bias/v
):'
2Adam/dense_429/kernel/v
": 2Adam/dense_429/bias/v
):'
2Adam/dense_430/kernel/v
": 2Adam/dense_430/bias/v
1:/2$Adam/batch_normalization_107/gamma/v
0:.2#Adam/batch_normalization_107/beta/v
(:&	2Adam/dense_431/kernel/v
!:2Adam/dense_431/bias/v
ò2ï
I__inference_model_107_layer_call_and_return_conditional_losses_1478344980
I__inference_model_107_layer_call_and_return_conditional_losses_1478345014
I__inference_model_107_layer_call_and_return_conditional_losses_1478345246
I__inference_model_107_layer_call_and_return_conditional_losses_1478345296À
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
é2æ
%__inference__wrapped_model_1478344678¼
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
annotationsª *,¢)
'$
	input_108ÿÿÿÿÿÿÿÿÿ4
2
.__inference_model_107_layer_call_fn_1478345141
.__inference_model_107_layer_call_fn_1478345078
.__inference_model_107_layer_call_fn_1478345325
.__inference_model_107_layer_call_fn_1478345354À
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
õ2ò
K__inference_flatten_107_layer_call_and_return_conditional_losses_1478345360¢
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
Ú2×
0__inference_flatten_107_layer_call_fn_1478345365¢
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
ó2ð
I__inference_dense_428_layer_call_and_return_conditional_losses_1478345376¢
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
.__inference_dense_428_layer_call_fn_1478345385¢
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
ó2ð
I__inference_dense_429_layer_call_and_return_conditional_losses_1478345396¢
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
.__inference_dense_429_layer_call_fn_1478345405¢
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
ó2ð
I__inference_dense_430_layer_call_and_return_conditional_losses_1478345416¢
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
.__inference_dense_430_layer_call_fn_1478345425¢
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
ì2é
W__inference_batch_normalization_107_layer_call_and_return_conditional_losses_1478345461
W__inference_batch_normalization_107_layer_call_and_return_conditional_losses_1478345481´
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
¶2³
<__inference_batch_normalization_107_layer_call_fn_1478345494
<__inference_batch_normalization_107_layer_call_fn_1478345507´
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
ó2ð
I__inference_dense_431_layer_call_and_return_conditional_losses_1478345518¢
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
.__inference_dense_431_layer_call_fn_1478345527¢
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
ÑBÎ
(__inference_signature_wrapper_1478345180	input_108"
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
 ¦
%__inference__wrapped_model_1478344678}(%'&-.6¢3
,¢)
'$
	input_108ÿÿÿÿÿÿÿÿÿ4
ª "5ª2
0
	dense_431# 
	dense_431ÿÿÿÿÿÿÿÿÿ¿
W__inference_batch_normalization_107_layer_call_and_return_conditional_losses_1478345461d'(%&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¿
W__inference_batch_normalization_107_layer_call_and_return_conditional_losses_1478345481d(%'&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
<__inference_batch_normalization_107_layer_call_fn_1478345494W'(%&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
<__inference_batch_normalization_107_layer_call_fn_1478345507W(%'&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ«
I__inference_dense_428_layer_call_and_return_conditional_losses_1478345376^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dense_428_layer_call_fn_1478345385Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ«
I__inference_dense_429_layer_call_and_return_conditional_losses_1478345396^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dense_429_layer_call_fn_1478345405Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ«
I__inference_dense_430_layer_call_and_return_conditional_losses_1478345416^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dense_430_layer_call_fn_1478345425Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
I__inference_dense_431_layer_call_and_return_conditional_losses_1478345518]-.0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dense_431_layer_call_fn_1478345527P-.0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¬
K__inference_flatten_107_layer_call_and_return_conditional_losses_1478345360]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ4
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_flatten_107_layer_call_fn_1478345365P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ4
ª "ÿÿÿÿÿÿÿÿÿÂ
I__inference_model_107_layer_call_and_return_conditional_losses_1478344980u'(%&-.>¢;
4¢1
'$
	input_108ÿÿÿÿÿÿÿÿÿ4
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
I__inference_model_107_layer_call_and_return_conditional_losses_1478345014u(%'&-.>¢;
4¢1
'$
	input_108ÿÿÿÿÿÿÿÿÿ4
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¿
I__inference_model_107_layer_call_and_return_conditional_losses_1478345246r'(%&-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¿
I__inference_model_107_layer_call_and_return_conditional_losses_1478345296r(%'&-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_model_107_layer_call_fn_1478345078h'(%&-.>¢;
4¢1
'$
	input_108ÿÿÿÿÿÿÿÿÿ4
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_model_107_layer_call_fn_1478345141h(%'&-.>¢;
4¢1
'$
	input_108ÿÿÿÿÿÿÿÿÿ4
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_model_107_layer_call_fn_1478345325e'(%&-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_model_107_layer_call_fn_1478345354e(%'&-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p 

 
ª "ÿÿÿÿÿÿÿÿÿ·
(__inference_signature_wrapper_1478345180(%'&-.C¢@
¢ 
9ª6
4
	input_108'$
	input_108ÿÿÿÿÿÿÿÿÿ4"5ª2
0
	dense_431# 
	dense_431ÿÿÿÿÿÿÿÿÿ