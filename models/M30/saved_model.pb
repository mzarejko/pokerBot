ñ

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
 "serve*2.4.12unknown8«ú
~
dense_260/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_260/kernel
w
$dense_260/kernel/Read/ReadVariableOpReadVariableOpdense_260/kernel* 
_output_shapes
:
*
dtype0
u
dense_260/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_260/bias
n
"dense_260/bias/Read/ReadVariableOpReadVariableOpdense_260/bias*
_output_shapes	
:*
dtype0
~
dense_261/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_261/kernel
w
$dense_261/kernel/Read/ReadVariableOpReadVariableOpdense_261/kernel* 
_output_shapes
:
*
dtype0
u
dense_261/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_261/bias
n
"dense_261/bias/Read/ReadVariableOpReadVariableOpdense_261/bias*
_output_shapes	
:*
dtype0
~
dense_262/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_262/kernel
w
$dense_262/kernel/Read/ReadVariableOpReadVariableOpdense_262/kernel* 
_output_shapes
:
*
dtype0
u
dense_262/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_262/bias
n
"dense_262/bias/Read/ReadVariableOpReadVariableOpdense_262/bias*
_output_shapes	
:*
dtype0

batch_normalization_65/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_65/gamma

0batch_normalization_65/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_65/gamma*
_output_shapes	
:*
dtype0

batch_normalization_65/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_65/beta

/batch_normalization_65/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_65/beta*
_output_shapes	
:*
dtype0

"batch_normalization_65/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_65/moving_mean

6batch_normalization_65/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_65/moving_mean*
_output_shapes	
:*
dtype0
¥
&batch_normalization_65/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_65/moving_variance

:batch_normalization_65/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_65/moving_variance*
_output_shapes	
:*
dtype0
}
dense_263/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_namedense_263/kernel
v
$dense_263/kernel/Read/ReadVariableOpReadVariableOpdense_263/kernel*
_output_shapes
:	*
dtype0
t
dense_263/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_263/bias
m
"dense_263/bias/Read/ReadVariableOpReadVariableOpdense_263/bias*
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
Adam/dense_260/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_260/kernel/m

+Adam/dense_260/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_260/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_260/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_260/bias/m
|
)Adam/dense_260/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_260/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_261/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_261/kernel/m

+Adam/dense_261/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_261/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_261/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_261/bias/m
|
)Adam/dense_261/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_261/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_262/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_262/kernel/m

+Adam/dense_262/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_262/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_262/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_262/bias/m
|
)Adam/dense_262/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_262/bias/m*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_65/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_65/gamma/m

7Adam/batch_normalization_65/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_65/gamma/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_65/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_65/beta/m

6Adam/batch_normalization_65/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_65/beta/m*
_output_shapes	
:*
dtype0

Adam/dense_263/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/dense_263/kernel/m

+Adam/dense_263/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_263/kernel/m*
_output_shapes
:	*
dtype0

Adam/dense_263/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_263/bias/m
{
)Adam/dense_263/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_263/bias/m*
_output_shapes
:*
dtype0

Adam/dense_260/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_260/kernel/v

+Adam/dense_260/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_260/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_260/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_260/bias/v
|
)Adam/dense_260/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_260/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_261/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_261/kernel/v

+Adam/dense_261/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_261/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_261/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_261/bias/v
|
)Adam/dense_261/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_261/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_262/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_262/kernel/v

+Adam/dense_262/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_262/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_262/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_262/bias/v
|
)Adam/dense_262/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_262/bias/v*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_65/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_65/gamma/v

7Adam/batch_normalization_65/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_65/gamma/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_65/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_65/beta/v

6Adam/batch_normalization_65/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_65/beta/v*
_output_shapes	
:*
dtype0

Adam/dense_263/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/dense_263/kernel/v

+Adam/dense_263/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_263/kernel/v*
_output_shapes
:	*
dtype0

Adam/dense_263/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_263/bias/v
{
)Adam/dense_263/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_263/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Ô:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*:
value:B: Bû9
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
VARIABLE_VALUEdense_260/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_260/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_261/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_261/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_262/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_262/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_65/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_65/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_65/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_65/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_263/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_263/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_260/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_260/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_261/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_261/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_262/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_262/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_65/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_65/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_263/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_263/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_260/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_260/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_261/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_261/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_262/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_262/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/batch_normalization_65/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_65/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_263/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_263/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_66Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ4
Ý
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_66dense_260/kerneldense_260/biasdense_261/kerneldense_261/biasdense_262/kerneldense_262/bias&batch_normalization_65/moving_variancebatch_normalization_65/gamma"batch_normalization_65/moving_meanbatch_normalization_65/betadense_263/kerneldense_263/bias*
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
'__inference_signature_wrapper_675710476
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Å
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_260/kernel/Read/ReadVariableOp"dense_260/bias/Read/ReadVariableOp$dense_261/kernel/Read/ReadVariableOp"dense_261/bias/Read/ReadVariableOp$dense_262/kernel/Read/ReadVariableOp"dense_262/bias/Read/ReadVariableOp0batch_normalization_65/gamma/Read/ReadVariableOp/batch_normalization_65/beta/Read/ReadVariableOp6batch_normalization_65/moving_mean/Read/ReadVariableOp:batch_normalization_65/moving_variance/Read/ReadVariableOp$dense_263/kernel/Read/ReadVariableOp"dense_263/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_260/kernel/m/Read/ReadVariableOp)Adam/dense_260/bias/m/Read/ReadVariableOp+Adam/dense_261/kernel/m/Read/ReadVariableOp)Adam/dense_261/bias/m/Read/ReadVariableOp+Adam/dense_262/kernel/m/Read/ReadVariableOp)Adam/dense_262/bias/m/Read/ReadVariableOp7Adam/batch_normalization_65/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_65/beta/m/Read/ReadVariableOp+Adam/dense_263/kernel/m/Read/ReadVariableOp)Adam/dense_263/bias/m/Read/ReadVariableOp+Adam/dense_260/kernel/v/Read/ReadVariableOp)Adam/dense_260/bias/v/Read/ReadVariableOp+Adam/dense_261/kernel/v/Read/ReadVariableOp)Adam/dense_261/bias/v/Read/ReadVariableOp+Adam/dense_262/kernel/v/Read/ReadVariableOp)Adam/dense_262/bias/v/Read/ReadVariableOp7Adam/batch_normalization_65/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_65/beta/v/Read/ReadVariableOp+Adam/dense_263/kernel/v/Read/ReadVariableOp)Adam/dense_263/bias/v/Read/ReadVariableOpConst*4
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
"__inference__traced_save_675710963
´	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_260/kerneldense_260/biasdense_261/kerneldense_261/biasdense_262/kerneldense_262/biasbatch_normalization_65/gammabatch_normalization_65/beta"batch_normalization_65/moving_mean&batch_normalization_65/moving_variancedense_263/kerneldense_263/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_260/kernel/mAdam/dense_260/bias/mAdam/dense_261/kernel/mAdam/dense_261/bias/mAdam/dense_262/kernel/mAdam/dense_262/bias/m#Adam/batch_normalization_65/gamma/m"Adam/batch_normalization_65/beta/mAdam/dense_263/kernel/mAdam/dense_263/bias/mAdam/dense_260/kernel/vAdam/dense_260/bias/vAdam/dense_261/kernel/vAdam/dense_261/bias/vAdam/dense_262/kernel/vAdam/dense_262/bias/v#Adam/batch_normalization_65/gamma/v"Adam/batch_normalization_65/beta/vAdam/dense_263/kernel/vAdam/dense_263/bias/v*3
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
%__inference__traced_restore_675711090öÐ
û	
á
H__inference_dense_260_layer_call_and_return_conditional_losses_675710143

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
ý	
á
H__inference_dense_263_layer_call_and_return_conditional_losses_675710259

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
­	

,__inference_model_65_layer_call_fn_675710650

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
G__inference_model_65_layer_call_and_return_conditional_losses_6757104102
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
ÿ!

G__inference_model_65_layer_call_and_return_conditional_losses_675710410

inputs
dense_260_675710380
dense_260_675710382
dense_261_675710385
dense_261_675710387
dense_262_675710390
dense_262_675710392$
 batch_normalization_65_675710395$
 batch_normalization_65_675710397$
 batch_normalization_65_675710399$
 batch_normalization_65_675710401
dense_263_675710404
dense_263_675710406
identity¢.batch_normalization_65/StatefulPartitionedCall¢!dense_260/StatefulPartitionedCall¢!dense_261/StatefulPartitionedCall¢!dense_262/StatefulPartitionedCall¢!dense_263/StatefulPartitionedCallÞ
flatten_65/PartitionedCallPartitionedCallinputs*
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
I__inference_flatten_65_layer_call_and_return_conditional_losses_6757101242
flatten_65/PartitionedCallÀ
!dense_260/StatefulPartitionedCallStatefulPartitionedCall#flatten_65/PartitionedCall:output:0dense_260_675710380dense_260_675710382*
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
GPU 2J 8 *Q
fLRJ
H__inference_dense_260_layer_call_and_return_conditional_losses_6757101432#
!dense_260/StatefulPartitionedCallÇ
!dense_261/StatefulPartitionedCallStatefulPartitionedCall*dense_260/StatefulPartitionedCall:output:0dense_261_675710385dense_261_675710387*
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
GPU 2J 8 *Q
fLRJ
H__inference_dense_261_layer_call_and_return_conditional_losses_6757101702#
!dense_261/StatefulPartitionedCallÇ
!dense_262/StatefulPartitionedCallStatefulPartitionedCall*dense_261/StatefulPartitionedCall:output:0dense_262_675710390dense_262_675710392*
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
GPU 2J 8 *Q
fLRJ
H__inference_dense_262_layer_call_and_return_conditional_losses_6757101972#
!dense_262/StatefulPartitionedCallÐ
.batch_normalization_65/StatefulPartitionedCallStatefulPartitionedCall*dense_262/StatefulPartitionedCall:output:0 batch_normalization_65_675710395 batch_normalization_65_675710397 batch_normalization_65_675710399 batch_normalization_65_675710401*
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
U__inference_batch_normalization_65_layer_call_and_return_conditional_losses_67571010320
.batch_normalization_65/StatefulPartitionedCallÓ
!dense_263/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_65/StatefulPartitionedCall:output:0dense_263_675710404dense_263_675710406*
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
GPU 2J 8 *Q
fLRJ
H__inference_dense_263_layer_call_and_return_conditional_losses_6757102592#
!dense_263/StatefulPartitionedCall¿
IdentityIdentity*dense_263/StatefulPartitionedCall:output:0/^batch_normalization_65/StatefulPartitionedCall"^dense_260/StatefulPartitionedCall"^dense_261/StatefulPartitionedCall"^dense_262/StatefulPartitionedCall"^dense_263/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2`
.batch_normalization_65/StatefulPartitionedCall.batch_normalization_65/StatefulPartitionedCall2F
!dense_260/StatefulPartitionedCall!dense_260/StatefulPartitionedCall2F
!dense_261/StatefulPartitionedCall!dense_261/StatefulPartitionedCall2F
!dense_262/StatefulPartitionedCall!dense_262/StatefulPartitionedCall2F
!dense_263/StatefulPartitionedCall!dense_263/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
û	
á
H__inference_dense_261_layer_call_and_return_conditional_losses_675710692

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
û	
á
H__inference_dense_261_layer_call_and_return_conditional_losses_675710170

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
¹
e
I__inference_flatten_65_layer_call_and_return_conditional_losses_675710656

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
û	
á
H__inference_dense_260_layer_call_and_return_conditional_losses_675710672

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
é

-__inference_dense_260_layer_call_fn_675710681

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
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_260_layer_call_and_return_conditional_losses_6757101432
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
Á
­
:__inference_batch_normalization_65_layer_call_fn_675710790

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
U__inference_batch_normalization_65_layer_call_and_return_conditional_losses_6757100702
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
Í0
Ò
U__inference_batch_normalization_65_layer_call_and_return_conditional_losses_675710070

inputs
assignmovingavg_675710045
assignmovingavg_1_675710051)
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
 loc:@AssignMovingAvg/675710045*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_675710045*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpõ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg/675710045*
_output_shapes	
:2
AssignMovingAvg/subì
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg/675710045*
_output_shapes	
:2
AssignMovingAvg/mulµ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_675710045AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg/675710045*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÕ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*.
_class$
" loc:@AssignMovingAvg_1/675710051*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_675710051*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÿ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/675710051*
_output_shapes	
:2
AssignMovingAvg_1/subö
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/675710051*
_output_shapes	
:2
AssignMovingAvg_1/mulÁ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_675710051AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*.
_class$
" loc:@AssignMovingAvg_1/675710051*
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
û	
á
H__inference_dense_262_layer_call_and_return_conditional_losses_675710712

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
Ã
­
:__inference_batch_normalization_65_layer_call_fn_675710803

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
U__inference_batch_normalization_65_layer_call_and_return_conditional_losses_6757101032
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
	

'__inference_signature_wrapper_675710476
input_66
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
StatefulPartitionedCallStatefulPartitionedCallinput_66unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
$__inference__wrapped_model_6757099742
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
input_66
¹
e
I__inference_flatten_65_layer_call_and_return_conditional_losses_675710124

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
¦
J
.__inference_flatten_65_layer_call_fn_675710661

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
I__inference_flatten_65_layer_call_and_return_conditional_losses_6757101242
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
ý!

G__inference_model_65_layer_call_and_return_conditional_losses_675710347

inputs
dense_260_675710317
dense_260_675710319
dense_261_675710322
dense_261_675710324
dense_262_675710327
dense_262_675710329$
 batch_normalization_65_675710332$
 batch_normalization_65_675710334$
 batch_normalization_65_675710336$
 batch_normalization_65_675710338
dense_263_675710341
dense_263_675710343
identity¢.batch_normalization_65/StatefulPartitionedCall¢!dense_260/StatefulPartitionedCall¢!dense_261/StatefulPartitionedCall¢!dense_262/StatefulPartitionedCall¢!dense_263/StatefulPartitionedCallÞ
flatten_65/PartitionedCallPartitionedCallinputs*
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
I__inference_flatten_65_layer_call_and_return_conditional_losses_6757101242
flatten_65/PartitionedCallÀ
!dense_260/StatefulPartitionedCallStatefulPartitionedCall#flatten_65/PartitionedCall:output:0dense_260_675710317dense_260_675710319*
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
GPU 2J 8 *Q
fLRJ
H__inference_dense_260_layer_call_and_return_conditional_losses_6757101432#
!dense_260/StatefulPartitionedCallÇ
!dense_261/StatefulPartitionedCallStatefulPartitionedCall*dense_260/StatefulPartitionedCall:output:0dense_261_675710322dense_261_675710324*
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
GPU 2J 8 *Q
fLRJ
H__inference_dense_261_layer_call_and_return_conditional_losses_6757101702#
!dense_261/StatefulPartitionedCallÇ
!dense_262/StatefulPartitionedCallStatefulPartitionedCall*dense_261/StatefulPartitionedCall:output:0dense_262_675710327dense_262_675710329*
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
GPU 2J 8 *Q
fLRJ
H__inference_dense_262_layer_call_and_return_conditional_losses_6757101972#
!dense_262/StatefulPartitionedCallÎ
.batch_normalization_65/StatefulPartitionedCallStatefulPartitionedCall*dense_262/StatefulPartitionedCall:output:0 batch_normalization_65_675710332 batch_normalization_65_675710334 batch_normalization_65_675710336 batch_normalization_65_675710338*
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
U__inference_batch_normalization_65_layer_call_and_return_conditional_losses_67571007020
.batch_normalization_65/StatefulPartitionedCallÓ
!dense_263/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_65/StatefulPartitionedCall:output:0dense_263_675710341dense_263_675710343*
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
GPU 2J 8 *Q
fLRJ
H__inference_dense_263_layer_call_and_return_conditional_losses_6757102592#
!dense_263/StatefulPartitionedCall¿
IdentityIdentity*dense_263/StatefulPartitionedCall:output:0/^batch_normalization_65/StatefulPartitionedCall"^dense_260/StatefulPartitionedCall"^dense_261/StatefulPartitionedCall"^dense_262/StatefulPartitionedCall"^dense_263/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2`
.batch_normalization_65/StatefulPartitionedCall.batch_normalization_65/StatefulPartitionedCall2F
!dense_260/StatefulPartitionedCall!dense_260/StatefulPartitionedCall2F
!dense_261/StatefulPartitionedCall!dense_261/StatefulPartitionedCall2F
!dense_262/StatefulPartitionedCall!dense_262/StatefulPartitionedCall2F
!dense_263/StatefulPartitionedCall!dense_263/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
æ

U__inference_batch_normalization_65_layer_call_and_return_conditional_losses_675710103

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
ç

-__inference_dense_263_layer_call_fn_675710823

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
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_263_layer_call_and_return_conditional_losses_6757102592
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
"

G__inference_model_65_layer_call_and_return_conditional_losses_675710310
input_66
dense_260_675710280
dense_260_675710282
dense_261_675710285
dense_261_675710287
dense_262_675710290
dense_262_675710292$
 batch_normalization_65_675710295$
 batch_normalization_65_675710297$
 batch_normalization_65_675710299$
 batch_normalization_65_675710301
dense_263_675710304
dense_263_675710306
identity¢.batch_normalization_65/StatefulPartitionedCall¢!dense_260/StatefulPartitionedCall¢!dense_261/StatefulPartitionedCall¢!dense_262/StatefulPartitionedCall¢!dense_263/StatefulPartitionedCallà
flatten_65/PartitionedCallPartitionedCallinput_66*
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
I__inference_flatten_65_layer_call_and_return_conditional_losses_6757101242
flatten_65/PartitionedCallÀ
!dense_260/StatefulPartitionedCallStatefulPartitionedCall#flatten_65/PartitionedCall:output:0dense_260_675710280dense_260_675710282*
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
GPU 2J 8 *Q
fLRJ
H__inference_dense_260_layer_call_and_return_conditional_losses_6757101432#
!dense_260/StatefulPartitionedCallÇ
!dense_261/StatefulPartitionedCallStatefulPartitionedCall*dense_260/StatefulPartitionedCall:output:0dense_261_675710285dense_261_675710287*
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
GPU 2J 8 *Q
fLRJ
H__inference_dense_261_layer_call_and_return_conditional_losses_6757101702#
!dense_261/StatefulPartitionedCallÇ
!dense_262/StatefulPartitionedCallStatefulPartitionedCall*dense_261/StatefulPartitionedCall:output:0dense_262_675710290dense_262_675710292*
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
GPU 2J 8 *Q
fLRJ
H__inference_dense_262_layer_call_and_return_conditional_losses_6757101972#
!dense_262/StatefulPartitionedCallÐ
.batch_normalization_65/StatefulPartitionedCallStatefulPartitionedCall*dense_262/StatefulPartitionedCall:output:0 batch_normalization_65_675710295 batch_normalization_65_675710297 batch_normalization_65_675710299 batch_normalization_65_675710301*
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
U__inference_batch_normalization_65_layer_call_and_return_conditional_losses_67571010320
.batch_normalization_65/StatefulPartitionedCallÓ
!dense_263/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_65/StatefulPartitionedCall:output:0dense_263_675710304dense_263_675710306*
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
GPU 2J 8 *Q
fLRJ
H__inference_dense_263_layer_call_and_return_conditional_losses_6757102592#
!dense_263/StatefulPartitionedCall¿
IdentityIdentity*dense_263/StatefulPartitionedCall:output:0/^batch_normalization_65/StatefulPartitionedCall"^dense_260/StatefulPartitionedCall"^dense_261/StatefulPartitionedCall"^dense_262/StatefulPartitionedCall"^dense_263/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2`
.batch_normalization_65/StatefulPartitionedCall.batch_normalization_65/StatefulPartitionedCall2F
!dense_260/StatefulPartitionedCall!dense_260/StatefulPartitionedCall2F
!dense_261/StatefulPartitionedCall!dense_261/StatefulPartitionedCall2F
!dense_262/StatefulPartitionedCall!dense_262/StatefulPartitionedCall2F
!dense_263/StatefulPartitionedCall!dense_263/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
input_66
"

G__inference_model_65_layer_call_and_return_conditional_losses_675710276
input_66
dense_260_675710154
dense_260_675710156
dense_261_675710181
dense_261_675710183
dense_262_675710208
dense_262_675710210$
 batch_normalization_65_675710239$
 batch_normalization_65_675710241$
 batch_normalization_65_675710243$
 batch_normalization_65_675710245
dense_263_675710270
dense_263_675710272
identity¢.batch_normalization_65/StatefulPartitionedCall¢!dense_260/StatefulPartitionedCall¢!dense_261/StatefulPartitionedCall¢!dense_262/StatefulPartitionedCall¢!dense_263/StatefulPartitionedCallà
flatten_65/PartitionedCallPartitionedCallinput_66*
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
I__inference_flatten_65_layer_call_and_return_conditional_losses_6757101242
flatten_65/PartitionedCallÀ
!dense_260/StatefulPartitionedCallStatefulPartitionedCall#flatten_65/PartitionedCall:output:0dense_260_675710154dense_260_675710156*
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
GPU 2J 8 *Q
fLRJ
H__inference_dense_260_layer_call_and_return_conditional_losses_6757101432#
!dense_260/StatefulPartitionedCallÇ
!dense_261/StatefulPartitionedCallStatefulPartitionedCall*dense_260/StatefulPartitionedCall:output:0dense_261_675710181dense_261_675710183*
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
GPU 2J 8 *Q
fLRJ
H__inference_dense_261_layer_call_and_return_conditional_losses_6757101702#
!dense_261/StatefulPartitionedCallÇ
!dense_262/StatefulPartitionedCallStatefulPartitionedCall*dense_261/StatefulPartitionedCall:output:0dense_262_675710208dense_262_675710210*
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
GPU 2J 8 *Q
fLRJ
H__inference_dense_262_layer_call_and_return_conditional_losses_6757101972#
!dense_262/StatefulPartitionedCallÎ
.batch_normalization_65/StatefulPartitionedCallStatefulPartitionedCall*dense_262/StatefulPartitionedCall:output:0 batch_normalization_65_675710239 batch_normalization_65_675710241 batch_normalization_65_675710243 batch_normalization_65_675710245*
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
U__inference_batch_normalization_65_layer_call_and_return_conditional_losses_67571007020
.batch_normalization_65/StatefulPartitionedCallÓ
!dense_263/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_65/StatefulPartitionedCall:output:0dense_263_675710270dense_263_675710272*
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
GPU 2J 8 *Q
fLRJ
H__inference_dense_263_layer_call_and_return_conditional_losses_6757102592#
!dense_263/StatefulPartitionedCall¿
IdentityIdentity*dense_263/StatefulPartitionedCall:output:0/^batch_normalization_65/StatefulPartitionedCall"^dense_260/StatefulPartitionedCall"^dense_261/StatefulPartitionedCall"^dense_262/StatefulPartitionedCall"^dense_263/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2`
.batch_normalization_65/StatefulPartitionedCall.batch_normalization_65/StatefulPartitionedCall2F
!dense_260/StatefulPartitionedCall!dense_260/StatefulPartitionedCall2F
!dense_261/StatefulPartitionedCall!dense_261/StatefulPartitionedCall2F
!dense_262/StatefulPartitionedCall!dense_262/StatefulPartitionedCall2F
!dense_263/StatefulPartitionedCall!dense_263/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
input_66
¥S
õ

$__inference__wrapped_model_675709974
input_665
1model_65_dense_260_matmul_readvariableop_resource6
2model_65_dense_260_biasadd_readvariableop_resource5
1model_65_dense_261_matmul_readvariableop_resource6
2model_65_dense_261_biasadd_readvariableop_resource5
1model_65_dense_262_matmul_readvariableop_resource6
2model_65_dense_262_biasadd_readvariableop_resourceE
Amodel_65_batch_normalization_65_batchnorm_readvariableop_resourceI
Emodel_65_batch_normalization_65_batchnorm_mul_readvariableop_resourceG
Cmodel_65_batch_normalization_65_batchnorm_readvariableop_1_resourceG
Cmodel_65_batch_normalization_65_batchnorm_readvariableop_2_resource5
1model_65_dense_263_matmul_readvariableop_resource6
2model_65_dense_263_biasadd_readvariableop_resource
identity¢8model_65/batch_normalization_65/batchnorm/ReadVariableOp¢:model_65/batch_normalization_65/batchnorm/ReadVariableOp_1¢:model_65/batch_normalization_65/batchnorm/ReadVariableOp_2¢<model_65/batch_normalization_65/batchnorm/mul/ReadVariableOp¢)model_65/dense_260/BiasAdd/ReadVariableOp¢(model_65/dense_260/MatMul/ReadVariableOp¢)model_65/dense_261/BiasAdd/ReadVariableOp¢(model_65/dense_261/MatMul/ReadVariableOp¢)model_65/dense_262/BiasAdd/ReadVariableOp¢(model_65/dense_262/MatMul/ReadVariableOp¢)model_65/dense_263/BiasAdd/ReadVariableOp¢(model_65/dense_263/MatMul/ReadVariableOp
model_65/flatten_65/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model_65/flatten_65/Const¦
model_65/flatten_65/ReshapeReshapeinput_66"model_65/flatten_65/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_65/flatten_65/ReshapeÈ
(model_65/dense_260/MatMul/ReadVariableOpReadVariableOp1model_65_dense_260_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02*
(model_65/dense_260/MatMul/ReadVariableOpË
model_65/dense_260/MatMulMatMul$model_65/flatten_65/Reshape:output:00model_65/dense_260/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_65/dense_260/MatMulÆ
)model_65/dense_260/BiasAdd/ReadVariableOpReadVariableOp2model_65_dense_260_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model_65/dense_260/BiasAdd/ReadVariableOpÎ
model_65/dense_260/BiasAddBiasAdd#model_65/dense_260/MatMul:product:01model_65/dense_260/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_65/dense_260/BiasAdd
model_65/dense_260/ReluRelu#model_65/dense_260/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_65/dense_260/ReluÈ
(model_65/dense_261/MatMul/ReadVariableOpReadVariableOp1model_65_dense_261_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02*
(model_65/dense_261/MatMul/ReadVariableOpÌ
model_65/dense_261/MatMulMatMul%model_65/dense_260/Relu:activations:00model_65/dense_261/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_65/dense_261/MatMulÆ
)model_65/dense_261/BiasAdd/ReadVariableOpReadVariableOp2model_65_dense_261_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model_65/dense_261/BiasAdd/ReadVariableOpÎ
model_65/dense_261/BiasAddBiasAdd#model_65/dense_261/MatMul:product:01model_65/dense_261/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_65/dense_261/BiasAdd
model_65/dense_261/ReluRelu#model_65/dense_261/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_65/dense_261/ReluÈ
(model_65/dense_262/MatMul/ReadVariableOpReadVariableOp1model_65_dense_262_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02*
(model_65/dense_262/MatMul/ReadVariableOpÌ
model_65/dense_262/MatMulMatMul%model_65/dense_261/Relu:activations:00model_65/dense_262/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_65/dense_262/MatMulÆ
)model_65/dense_262/BiasAdd/ReadVariableOpReadVariableOp2model_65_dense_262_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)model_65/dense_262/BiasAdd/ReadVariableOpÎ
model_65/dense_262/BiasAddBiasAdd#model_65/dense_262/MatMul:product:01model_65/dense_262/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_65/dense_262/BiasAdd
model_65/dense_262/ReluRelu#model_65/dense_262/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_65/dense_262/Reluó
8model_65/batch_normalization_65/batchnorm/ReadVariableOpReadVariableOpAmodel_65_batch_normalization_65_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02:
8model_65/batch_normalization_65/batchnorm/ReadVariableOp§
/model_65/batch_normalization_65/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:21
/model_65/batch_normalization_65/batchnorm/add/y
-model_65/batch_normalization_65/batchnorm/addAddV2@model_65/batch_normalization_65/batchnorm/ReadVariableOp:value:08model_65/batch_normalization_65/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2/
-model_65/batch_normalization_65/batchnorm/addÄ
/model_65/batch_normalization_65/batchnorm/RsqrtRsqrt1model_65/batch_normalization_65/batchnorm/add:z:0*
T0*
_output_shapes	
:21
/model_65/batch_normalization_65/batchnorm/Rsqrtÿ
<model_65/batch_normalization_65/batchnorm/mul/ReadVariableOpReadVariableOpEmodel_65_batch_normalization_65_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02>
<model_65/batch_normalization_65/batchnorm/mul/ReadVariableOp
-model_65/batch_normalization_65/batchnorm/mulMul3model_65/batch_normalization_65/batchnorm/Rsqrt:y:0Dmodel_65/batch_normalization_65/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2/
-model_65/batch_normalization_65/batchnorm/mulö
/model_65/batch_normalization_65/batchnorm/mul_1Mul%model_65/dense_262/Relu:activations:01model_65/batch_normalization_65/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/model_65/batch_normalization_65/batchnorm/mul_1ù
:model_65/batch_normalization_65/batchnorm/ReadVariableOp_1ReadVariableOpCmodel_65_batch_normalization_65_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02<
:model_65/batch_normalization_65/batchnorm/ReadVariableOp_1
/model_65/batch_normalization_65/batchnorm/mul_2MulBmodel_65/batch_normalization_65/batchnorm/ReadVariableOp_1:value:01model_65/batch_normalization_65/batchnorm/mul:z:0*
T0*
_output_shapes	
:21
/model_65/batch_normalization_65/batchnorm/mul_2ù
:model_65/batch_normalization_65/batchnorm/ReadVariableOp_2ReadVariableOpCmodel_65_batch_normalization_65_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02<
:model_65/batch_normalization_65/batchnorm/ReadVariableOp_2
-model_65/batch_normalization_65/batchnorm/subSubBmodel_65/batch_normalization_65/batchnorm/ReadVariableOp_2:value:03model_65/batch_normalization_65/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2/
-model_65/batch_normalization_65/batchnorm/sub
/model_65/batch_normalization_65/batchnorm/add_1AddV23model_65/batch_normalization_65/batchnorm/mul_1:z:01model_65/batch_normalization_65/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/model_65/batch_normalization_65/batchnorm/add_1Ç
(model_65/dense_263/MatMul/ReadVariableOpReadVariableOp1model_65_dense_263_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(model_65/dense_263/MatMul/ReadVariableOpÙ
model_65/dense_263/MatMulMatMul3model_65/batch_normalization_65/batchnorm/add_1:z:00model_65/dense_263/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_65/dense_263/MatMulÅ
)model_65/dense_263/BiasAdd/ReadVariableOpReadVariableOp2model_65_dense_263_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_65/dense_263/BiasAdd/ReadVariableOpÍ
model_65/dense_263/BiasAddBiasAdd#model_65/dense_263/MatMul:product:01model_65/dense_263/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_65/dense_263/BiasAdd
model_65/dense_263/SoftmaxSoftmax#model_65/dense_263/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_65/dense_263/SoftmaxÈ
IdentityIdentity$model_65/dense_263/Softmax:softmax:09^model_65/batch_normalization_65/batchnorm/ReadVariableOp;^model_65/batch_normalization_65/batchnorm/ReadVariableOp_1;^model_65/batch_normalization_65/batchnorm/ReadVariableOp_2=^model_65/batch_normalization_65/batchnorm/mul/ReadVariableOp*^model_65/dense_260/BiasAdd/ReadVariableOp)^model_65/dense_260/MatMul/ReadVariableOp*^model_65/dense_261/BiasAdd/ReadVariableOp)^model_65/dense_261/MatMul/ReadVariableOp*^model_65/dense_262/BiasAdd/ReadVariableOp)^model_65/dense_262/MatMul/ReadVariableOp*^model_65/dense_263/BiasAdd/ReadVariableOp)^model_65/dense_263/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2t
8model_65/batch_normalization_65/batchnorm/ReadVariableOp8model_65/batch_normalization_65/batchnorm/ReadVariableOp2x
:model_65/batch_normalization_65/batchnorm/ReadVariableOp_1:model_65/batch_normalization_65/batchnorm/ReadVariableOp_12x
:model_65/batch_normalization_65/batchnorm/ReadVariableOp_2:model_65/batch_normalization_65/batchnorm/ReadVariableOp_22|
<model_65/batch_normalization_65/batchnorm/mul/ReadVariableOp<model_65/batch_normalization_65/batchnorm/mul/ReadVariableOp2V
)model_65/dense_260/BiasAdd/ReadVariableOp)model_65/dense_260/BiasAdd/ReadVariableOp2T
(model_65/dense_260/MatMul/ReadVariableOp(model_65/dense_260/MatMul/ReadVariableOp2V
)model_65/dense_261/BiasAdd/ReadVariableOp)model_65/dense_261/BiasAdd/ReadVariableOp2T
(model_65/dense_261/MatMul/ReadVariableOp(model_65/dense_261/MatMul/ReadVariableOp2V
)model_65/dense_262/BiasAdd/ReadVariableOp)model_65/dense_262/BiasAdd/ReadVariableOp2T
(model_65/dense_262/MatMul/ReadVariableOp(model_65/dense_262/MatMul/ReadVariableOp2V
)model_65/dense_263/BiasAdd/ReadVariableOp)model_65/dense_263/BiasAdd/ReadVariableOp2T
(model_65/dense_263/MatMul/ReadVariableOp(model_65/dense_263/MatMul/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
"
_user_specified_name
input_66
Þo
²

G__inference_model_65_layer_call_and_return_conditional_losses_675710542

inputs,
(dense_260_matmul_readvariableop_resource-
)dense_260_biasadd_readvariableop_resource,
(dense_261_matmul_readvariableop_resource-
)dense_261_biasadd_readvariableop_resource,
(dense_262_matmul_readvariableop_resource-
)dense_262_biasadd_readvariableop_resource4
0batch_normalization_65_assignmovingavg_6757105106
2batch_normalization_65_assignmovingavg_1_675710516@
<batch_normalization_65_batchnorm_mul_readvariableop_resource<
8batch_normalization_65_batchnorm_readvariableop_resource,
(dense_263_matmul_readvariableop_resource-
)dense_263_biasadd_readvariableop_resource
identity¢:batch_normalization_65/AssignMovingAvg/AssignSubVariableOp¢5batch_normalization_65/AssignMovingAvg/ReadVariableOp¢<batch_normalization_65/AssignMovingAvg_1/AssignSubVariableOp¢7batch_normalization_65/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_65/batchnorm/ReadVariableOp¢3batch_normalization_65/batchnorm/mul/ReadVariableOp¢ dense_260/BiasAdd/ReadVariableOp¢dense_260/MatMul/ReadVariableOp¢ dense_261/BiasAdd/ReadVariableOp¢dense_261/MatMul/ReadVariableOp¢ dense_262/BiasAdd/ReadVariableOp¢dense_262/MatMul/ReadVariableOp¢ dense_263/BiasAdd/ReadVariableOp¢dense_263/MatMul/ReadVariableOpu
flatten_65/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_65/Const
flatten_65/ReshapeReshapeinputsflatten_65/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_65/Reshape­
dense_260/MatMul/ReadVariableOpReadVariableOp(dense_260_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_260/MatMul/ReadVariableOp§
dense_260/MatMulMatMulflatten_65/Reshape:output:0'dense_260/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_260/MatMul«
 dense_260/BiasAdd/ReadVariableOpReadVariableOp)dense_260_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_260/BiasAdd/ReadVariableOpª
dense_260/BiasAddBiasAdddense_260/MatMul:product:0(dense_260/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_260/BiasAddw
dense_260/ReluReludense_260/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_260/Relu­
dense_261/MatMul/ReadVariableOpReadVariableOp(dense_261_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_261/MatMul/ReadVariableOp¨
dense_261/MatMulMatMuldense_260/Relu:activations:0'dense_261/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_261/MatMul«
 dense_261/BiasAdd/ReadVariableOpReadVariableOp)dense_261_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_261/BiasAdd/ReadVariableOpª
dense_261/BiasAddBiasAdddense_261/MatMul:product:0(dense_261/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_261/BiasAddw
dense_261/ReluReludense_261/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_261/Relu­
dense_262/MatMul/ReadVariableOpReadVariableOp(dense_262_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_262/MatMul/ReadVariableOp¨
dense_262/MatMulMatMuldense_261/Relu:activations:0'dense_262/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_262/MatMul«
 dense_262/BiasAdd/ReadVariableOpReadVariableOp)dense_262_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_262/BiasAdd/ReadVariableOpª
dense_262/BiasAddBiasAdddense_262/MatMul:product:0(dense_262/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_262/BiasAddw
dense_262/ReluReludense_262/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_262/Relu¸
5batch_normalization_65/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_65/moments/mean/reduction_indicesë
#batch_normalization_65/moments/meanMeandense_262/Relu:activations:0>batch_normalization_65/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2%
#batch_normalization_65/moments/meanÂ
+batch_normalization_65/moments/StopGradientStopGradient,batch_normalization_65/moments/mean:output:0*
T0*
_output_shapes
:	2-
+batch_normalization_65/moments/StopGradient
0batch_normalization_65/moments/SquaredDifferenceSquaredDifferencedense_262/Relu:activations:04batch_normalization_65/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_65/moments/SquaredDifferenceÀ
9batch_normalization_65/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_65/moments/variance/reduction_indices
'batch_normalization_65/moments/varianceMean4batch_normalization_65/moments/SquaredDifference:z:0Bbatch_normalization_65/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2)
'batch_normalization_65/moments/varianceÆ
&batch_normalization_65/moments/SqueezeSqueeze,batch_normalization_65/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2(
&batch_normalization_65/moments/SqueezeÎ
(batch_normalization_65/moments/Squeeze_1Squeeze0batch_normalization_65/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2*
(batch_normalization_65/moments/Squeeze_1
,batch_normalization_65/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_65/AssignMovingAvg/675710510*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_65/AssignMovingAvg/decayÜ
5batch_normalization_65/AssignMovingAvg/ReadVariableOpReadVariableOp0batch_normalization_65_assignmovingavg_675710510*
_output_shapes	
:*
dtype027
5batch_normalization_65/AssignMovingAvg/ReadVariableOpè
*batch_normalization_65/AssignMovingAvg/subSub=batch_normalization_65/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_65/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_65/AssignMovingAvg/675710510*
_output_shapes	
:2,
*batch_normalization_65/AssignMovingAvg/subß
*batch_normalization_65/AssignMovingAvg/mulMul.batch_normalization_65/AssignMovingAvg/sub:z:05batch_normalization_65/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@batch_normalization_65/AssignMovingAvg/675710510*
_output_shapes	
:2,
*batch_normalization_65/AssignMovingAvg/mul¿
:batch_normalization_65/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp0batch_normalization_65_assignmovingavg_675710510.batch_normalization_65/AssignMovingAvg/mul:z:06^batch_normalization_65/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*C
_class9
75loc:@batch_normalization_65/AssignMovingAvg/675710510*
_output_shapes
 *
dtype02<
:batch_normalization_65/AssignMovingAvg/AssignSubVariableOp
.batch_normalization_65/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*E
_class;
97loc:@batch_normalization_65/AssignMovingAvg_1/675710516*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_65/AssignMovingAvg_1/decayâ
7batch_normalization_65/AssignMovingAvg_1/ReadVariableOpReadVariableOp2batch_normalization_65_assignmovingavg_1_675710516*
_output_shapes	
:*
dtype029
7batch_normalization_65/AssignMovingAvg_1/ReadVariableOpò
,batch_normalization_65/AssignMovingAvg_1/subSub?batch_normalization_65/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_65/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@batch_normalization_65/AssignMovingAvg_1/675710516*
_output_shapes	
:2.
,batch_normalization_65/AssignMovingAvg_1/subé
,batch_normalization_65/AssignMovingAvg_1/mulMul0batch_normalization_65/AssignMovingAvg_1/sub:z:07batch_normalization_65/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@batch_normalization_65/AssignMovingAvg_1/675710516*
_output_shapes	
:2.
,batch_normalization_65/AssignMovingAvg_1/mulË
<batch_normalization_65/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp2batch_normalization_65_assignmovingavg_1_6757105160batch_normalization_65/AssignMovingAvg_1/mul:z:08^batch_normalization_65/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*E
_class;
97loc:@batch_normalization_65/AssignMovingAvg_1/675710516*
_output_shapes
 *
dtype02>
<batch_normalization_65/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_65/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_65/batchnorm/add/yß
$batch_normalization_65/batchnorm/addAddV21batch_normalization_65/moments/Squeeze_1:output:0/batch_normalization_65/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2&
$batch_normalization_65/batchnorm/add©
&batch_normalization_65/batchnorm/RsqrtRsqrt(batch_normalization_65/batchnorm/add:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_65/batchnorm/Rsqrtä
3batch_normalization_65/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_65_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype025
3batch_normalization_65/batchnorm/mul/ReadVariableOpâ
$batch_normalization_65/batchnorm/mulMul*batch_normalization_65/batchnorm/Rsqrt:y:0;batch_normalization_65/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2&
$batch_normalization_65/batchnorm/mulÒ
&batch_normalization_65/batchnorm/mul_1Muldense_262/Relu:activations:0(batch_normalization_65/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_65/batchnorm/mul_1Ø
&batch_normalization_65/batchnorm/mul_2Mul/batch_normalization_65/moments/Squeeze:output:0(batch_normalization_65/batchnorm/mul:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_65/batchnorm/mul_2Ø
/batch_normalization_65/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_65_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype021
/batch_normalization_65/batchnorm/ReadVariableOpÞ
$batch_normalization_65/batchnorm/subSub7batch_normalization_65/batchnorm/ReadVariableOp:value:0*batch_normalization_65/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2&
$batch_normalization_65/batchnorm/subâ
&batch_normalization_65/batchnorm/add_1AddV2*batch_normalization_65/batchnorm/mul_1:z:0(batch_normalization_65/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_65/batchnorm/add_1¬
dense_263/MatMul/ReadVariableOpReadVariableOp(dense_263_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02!
dense_263/MatMul/ReadVariableOpµ
dense_263/MatMulMatMul*batch_normalization_65/batchnorm/add_1:z:0'dense_263/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_263/MatMulª
 dense_263/BiasAdd/ReadVariableOpReadVariableOp)dense_263_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_263/BiasAdd/ReadVariableOp©
dense_263/BiasAddBiasAdddense_263/MatMul:product:0(dense_263/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_263/BiasAdd
dense_263/SoftmaxSoftmaxdense_263/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_263/SoftmaxÙ
IdentityIdentitydense_263/Softmax:softmax:0;^batch_normalization_65/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_65/AssignMovingAvg/ReadVariableOp=^batch_normalization_65/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_65/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_65/batchnorm/ReadVariableOp4^batch_normalization_65/batchnorm/mul/ReadVariableOp!^dense_260/BiasAdd/ReadVariableOp ^dense_260/MatMul/ReadVariableOp!^dense_261/BiasAdd/ReadVariableOp ^dense_261/MatMul/ReadVariableOp!^dense_262/BiasAdd/ReadVariableOp ^dense_262/MatMul/ReadVariableOp!^dense_263/BiasAdd/ReadVariableOp ^dense_263/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2x
:batch_normalization_65/AssignMovingAvg/AssignSubVariableOp:batch_normalization_65/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_65/AssignMovingAvg/ReadVariableOp5batch_normalization_65/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_65/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_65/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_65/AssignMovingAvg_1/ReadVariableOp7batch_normalization_65/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_65/batchnorm/ReadVariableOp/batch_normalization_65/batchnorm/ReadVariableOp2j
3batch_normalization_65/batchnorm/mul/ReadVariableOp3batch_normalization_65/batchnorm/mul/ReadVariableOp2D
 dense_260/BiasAdd/ReadVariableOp dense_260/BiasAdd/ReadVariableOp2B
dense_260/MatMul/ReadVariableOpdense_260/MatMul/ReadVariableOp2D
 dense_261/BiasAdd/ReadVariableOp dense_261/BiasAdd/ReadVariableOp2B
dense_261/MatMul/ReadVariableOpdense_261/MatMul/ReadVariableOp2D
 dense_262/BiasAdd/ReadVariableOp dense_262/BiasAdd/ReadVariableOp2B
dense_262/MatMul/ReadVariableOpdense_262/MatMul/ReadVariableOp2D
 dense_263/BiasAdd/ReadVariableOp dense_263/BiasAdd/ReadVariableOp2B
dense_263/MatMul/ReadVariableOpdense_263/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs
û	
á
H__inference_dense_262_layer_call_and_return_conditional_losses_675710197

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
é

-__inference_dense_261_layer_call_fn_675710701

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
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_261_layer_call_and_return_conditional_losses_6757101702
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
é

-__inference_dense_262_layer_call_fn_675710721

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
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_262_layer_call_and_return_conditional_losses_6757101972
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
§
ª
%__inference__traced_restore_675711090
file_prefix%
!assignvariableop_dense_260_kernel%
!assignvariableop_1_dense_260_bias'
#assignvariableop_2_dense_261_kernel%
!assignvariableop_3_dense_261_bias'
#assignvariableop_4_dense_262_kernel%
!assignvariableop_5_dense_262_bias3
/assignvariableop_6_batch_normalization_65_gamma2
.assignvariableop_7_batch_normalization_65_beta9
5assignvariableop_8_batch_normalization_65_moving_mean=
9assignvariableop_9_batch_normalization_65_moving_variance(
$assignvariableop_10_dense_263_kernel&
"assignvariableop_11_dense_263_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count/
+assignvariableop_19_adam_dense_260_kernel_m-
)assignvariableop_20_adam_dense_260_bias_m/
+assignvariableop_21_adam_dense_261_kernel_m-
)assignvariableop_22_adam_dense_261_bias_m/
+assignvariableop_23_adam_dense_262_kernel_m-
)assignvariableop_24_adam_dense_262_bias_m;
7assignvariableop_25_adam_batch_normalization_65_gamma_m:
6assignvariableop_26_adam_batch_normalization_65_beta_m/
+assignvariableop_27_adam_dense_263_kernel_m-
)assignvariableop_28_adam_dense_263_bias_m/
+assignvariableop_29_adam_dense_260_kernel_v-
)assignvariableop_30_adam_dense_260_bias_v/
+assignvariableop_31_adam_dense_261_kernel_v-
)assignvariableop_32_adam_dense_261_bias_v/
+assignvariableop_33_adam_dense_262_kernel_v-
)assignvariableop_34_adam_dense_262_bias_v;
7assignvariableop_35_adam_batch_normalization_65_gamma_v:
6assignvariableop_36_adam_batch_normalization_65_beta_v/
+assignvariableop_37_adam_dense_263_kernel_v-
)assignvariableop_38_adam_dense_263_bias_v
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
AssignVariableOpAssignVariableOp!assignvariableop_dense_260_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_260_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_261_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_261_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_262_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_262_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6´
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_65_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7³
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_65_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8º
AssignVariableOp_8AssignVariableOp5assignvariableop_8_batch_normalization_65_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¾
AssignVariableOp_9AssignVariableOp9assignvariableop_9_batch_normalization_65_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¬
AssignVariableOp_10AssignVariableOp$assignvariableop_10_dense_263_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ª
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_263_biasIdentity_11:output:0"/device:CPU:0*
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
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_dense_260_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20±
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_260_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21³
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_261_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22±
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_261_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23³
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_262_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24±
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_262_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¿
AssignVariableOp_25AssignVariableOp7assignvariableop_25_adam_batch_normalization_65_gamma_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¾
AssignVariableOp_26AssignVariableOp6assignvariableop_26_adam_batch_normalization_65_beta_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27³
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_263_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28±
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_263_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29³
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_260_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30±
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_260_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31³
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_261_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32±
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_261_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33³
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_dense_262_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34±
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_262_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¿
AssignVariableOp_35AssignVariableOp7assignvariableop_35_adam_batch_normalization_65_gamma_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¾
AssignVariableOp_36AssignVariableOp6assignvariableop_36_adam_batch_normalization_65_beta_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37³
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_dense_263_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38±
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_dense_263_bias_vIdentity_38:output:0"/device:CPU:0*
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
Í0
Ò
U__inference_batch_normalization_65_layer_call_and_return_conditional_losses_675710757

inputs
assignmovingavg_675710732
assignmovingavg_1_675710738)
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
 loc:@AssignMovingAvg/675710732*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_675710732*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpõ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg/675710732*
_output_shapes	
:2
AssignMovingAvg/subì
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@AssignMovingAvg/675710732*
_output_shapes	
:2
AssignMovingAvg/mulµ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_675710732AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*,
_class"
 loc:@AssignMovingAvg/675710732*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÕ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*.
_class$
" loc:@AssignMovingAvg_1/675710738*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_675710738*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÿ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/675710738*
_output_shapes	
:2
AssignMovingAvg_1/subö
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*.
_class$
" loc:@AssignMovingAvg_1/675710738*
_output_shapes	
:2
AssignMovingAvg_1/mulÁ
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_675710738AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*.
_class$
" loc:@AssignMovingAvg_1/675710738*
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
U

"__inference__traced_save_675710963
file_prefix/
+savev2_dense_260_kernel_read_readvariableop-
)savev2_dense_260_bias_read_readvariableop/
+savev2_dense_261_kernel_read_readvariableop-
)savev2_dense_261_bias_read_readvariableop/
+savev2_dense_262_kernel_read_readvariableop-
)savev2_dense_262_bias_read_readvariableop;
7savev2_batch_normalization_65_gamma_read_readvariableop:
6savev2_batch_normalization_65_beta_read_readvariableopA
=savev2_batch_normalization_65_moving_mean_read_readvariableopE
Asavev2_batch_normalization_65_moving_variance_read_readvariableop/
+savev2_dense_263_kernel_read_readvariableop-
)savev2_dense_263_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_260_kernel_m_read_readvariableop4
0savev2_adam_dense_260_bias_m_read_readvariableop6
2savev2_adam_dense_261_kernel_m_read_readvariableop4
0savev2_adam_dense_261_bias_m_read_readvariableop6
2savev2_adam_dense_262_kernel_m_read_readvariableop4
0savev2_adam_dense_262_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_65_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_65_beta_m_read_readvariableop6
2savev2_adam_dense_263_kernel_m_read_readvariableop4
0savev2_adam_dense_263_bias_m_read_readvariableop6
2savev2_adam_dense_260_kernel_v_read_readvariableop4
0savev2_adam_dense_260_bias_v_read_readvariableop6
2savev2_adam_dense_261_kernel_v_read_readvariableop4
0savev2_adam_dense_261_bias_v_read_readvariableop6
2savev2_adam_dense_262_kernel_v_read_readvariableop4
0savev2_adam_dense_262_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_65_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_65_beta_v_read_readvariableop6
2savev2_adam_dense_263_kernel_v_read_readvariableop4
0savev2_adam_dense_263_bias_v_read_readvariableop
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
SaveV2/shape_and_slicesò
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_260_kernel_read_readvariableop)savev2_dense_260_bias_read_readvariableop+savev2_dense_261_kernel_read_readvariableop)savev2_dense_261_bias_read_readvariableop+savev2_dense_262_kernel_read_readvariableop)savev2_dense_262_bias_read_readvariableop7savev2_batch_normalization_65_gamma_read_readvariableop6savev2_batch_normalization_65_beta_read_readvariableop=savev2_batch_normalization_65_moving_mean_read_readvariableopAsavev2_batch_normalization_65_moving_variance_read_readvariableop+savev2_dense_263_kernel_read_readvariableop)savev2_dense_263_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_260_kernel_m_read_readvariableop0savev2_adam_dense_260_bias_m_read_readvariableop2savev2_adam_dense_261_kernel_m_read_readvariableop0savev2_adam_dense_261_bias_m_read_readvariableop2savev2_adam_dense_262_kernel_m_read_readvariableop0savev2_adam_dense_262_bias_m_read_readvariableop>savev2_adam_batch_normalization_65_gamma_m_read_readvariableop=savev2_adam_batch_normalization_65_beta_m_read_readvariableop2savev2_adam_dense_263_kernel_m_read_readvariableop0savev2_adam_dense_263_bias_m_read_readvariableop2savev2_adam_dense_260_kernel_v_read_readvariableop0savev2_adam_dense_260_bias_v_read_readvariableop2savev2_adam_dense_261_kernel_v_read_readvariableop0savev2_adam_dense_261_bias_v_read_readvariableop2savev2_adam_dense_262_kernel_v_read_readvariableop0savev2_adam_dense_262_bias_v_read_readvariableop>savev2_adam_batch_normalization_65_gamma_v_read_readvariableop=savev2_adam_batch_normalization_65_beta_v_read_readvariableop2savev2_adam_dense_263_kernel_v_read_readvariableop0savev2_adam_dense_263_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
æ

U__inference_batch_normalization_65_layer_call_and_return_conditional_losses_675710777

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
«	

,__inference_model_65_layer_call_fn_675710621

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
G__inference_model_65_layer_call_and_return_conditional_losses_6757103472
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
±	

,__inference_model_65_layer_call_fn_675710374
input_66
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
StatefulPartitionedCallStatefulPartitionedCallinput_66unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
G__inference_model_65_layer_call_and_return_conditional_losses_6757103472
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
input_66
³	

,__inference_model_65_layer_call_fn_675710437
input_66
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
StatefulPartitionedCallStatefulPartitionedCallinput_66unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
G__inference_model_65_layer_call_and_return_conditional_losses_6757104102
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
input_66
ý	
á
H__inference_dense_263_layer_call_and_return_conditional_losses_675710814

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
G
¾	
G__inference_model_65_layer_call_and_return_conditional_losses_675710592

inputs,
(dense_260_matmul_readvariableop_resource-
)dense_260_biasadd_readvariableop_resource,
(dense_261_matmul_readvariableop_resource-
)dense_261_biasadd_readvariableop_resource,
(dense_262_matmul_readvariableop_resource-
)dense_262_biasadd_readvariableop_resource<
8batch_normalization_65_batchnorm_readvariableop_resource@
<batch_normalization_65_batchnorm_mul_readvariableop_resource>
:batch_normalization_65_batchnorm_readvariableop_1_resource>
:batch_normalization_65_batchnorm_readvariableop_2_resource,
(dense_263_matmul_readvariableop_resource-
)dense_263_biasadd_readvariableop_resource
identity¢/batch_normalization_65/batchnorm/ReadVariableOp¢1batch_normalization_65/batchnorm/ReadVariableOp_1¢1batch_normalization_65/batchnorm/ReadVariableOp_2¢3batch_normalization_65/batchnorm/mul/ReadVariableOp¢ dense_260/BiasAdd/ReadVariableOp¢dense_260/MatMul/ReadVariableOp¢ dense_261/BiasAdd/ReadVariableOp¢dense_261/MatMul/ReadVariableOp¢ dense_262/BiasAdd/ReadVariableOp¢dense_262/MatMul/ReadVariableOp¢ dense_263/BiasAdd/ReadVariableOp¢dense_263/MatMul/ReadVariableOpu
flatten_65/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_65/Const
flatten_65/ReshapeReshapeinputsflatten_65/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_65/Reshape­
dense_260/MatMul/ReadVariableOpReadVariableOp(dense_260_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_260/MatMul/ReadVariableOp§
dense_260/MatMulMatMulflatten_65/Reshape:output:0'dense_260/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_260/MatMul«
 dense_260/BiasAdd/ReadVariableOpReadVariableOp)dense_260_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_260/BiasAdd/ReadVariableOpª
dense_260/BiasAddBiasAdddense_260/MatMul:product:0(dense_260/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_260/BiasAddw
dense_260/ReluReludense_260/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_260/Relu­
dense_261/MatMul/ReadVariableOpReadVariableOp(dense_261_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_261/MatMul/ReadVariableOp¨
dense_261/MatMulMatMuldense_260/Relu:activations:0'dense_261/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_261/MatMul«
 dense_261/BiasAdd/ReadVariableOpReadVariableOp)dense_261_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_261/BiasAdd/ReadVariableOpª
dense_261/BiasAddBiasAdddense_261/MatMul:product:0(dense_261/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_261/BiasAddw
dense_261/ReluReludense_261/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_261/Relu­
dense_262/MatMul/ReadVariableOpReadVariableOp(dense_262_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_262/MatMul/ReadVariableOp¨
dense_262/MatMulMatMuldense_261/Relu:activations:0'dense_262/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_262/MatMul«
 dense_262/BiasAdd/ReadVariableOpReadVariableOp)dense_262_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_262/BiasAdd/ReadVariableOpª
dense_262/BiasAddBiasAdddense_262/MatMul:product:0(dense_262/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_262/BiasAddw
dense_262/ReluReludense_262/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_262/ReluØ
/batch_normalization_65/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_65_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype021
/batch_normalization_65/batchnorm/ReadVariableOp
&batch_normalization_65/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_65/batchnorm/add/yå
$batch_normalization_65/batchnorm/addAddV27batch_normalization_65/batchnorm/ReadVariableOp:value:0/batch_normalization_65/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2&
$batch_normalization_65/batchnorm/add©
&batch_normalization_65/batchnorm/RsqrtRsqrt(batch_normalization_65/batchnorm/add:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_65/batchnorm/Rsqrtä
3batch_normalization_65/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_65_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype025
3batch_normalization_65/batchnorm/mul/ReadVariableOpâ
$batch_normalization_65/batchnorm/mulMul*batch_normalization_65/batchnorm/Rsqrt:y:0;batch_normalization_65/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2&
$batch_normalization_65/batchnorm/mulÒ
&batch_normalization_65/batchnorm/mul_1Muldense_262/Relu:activations:0(batch_normalization_65/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_65/batchnorm/mul_1Þ
1batch_normalization_65/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_65_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype023
1batch_normalization_65/batchnorm/ReadVariableOp_1â
&batch_normalization_65/batchnorm/mul_2Mul9batch_normalization_65/batchnorm/ReadVariableOp_1:value:0(batch_normalization_65/batchnorm/mul:z:0*
T0*
_output_shapes	
:2(
&batch_normalization_65/batchnorm/mul_2Þ
1batch_normalization_65/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_65_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype023
1batch_normalization_65/batchnorm/ReadVariableOp_2à
$batch_normalization_65/batchnorm/subSub9batch_normalization_65/batchnorm/ReadVariableOp_2:value:0*batch_normalization_65/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2&
$batch_normalization_65/batchnorm/subâ
&batch_normalization_65/batchnorm/add_1AddV2*batch_normalization_65/batchnorm/mul_1:z:0(batch_normalization_65/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_65/batchnorm/add_1¬
dense_263/MatMul/ReadVariableOpReadVariableOp(dense_263_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02!
dense_263/MatMul/ReadVariableOpµ
dense_263/MatMulMatMul*batch_normalization_65/batchnorm/add_1:z:0'dense_263/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_263/MatMulª
 dense_263/BiasAdd/ReadVariableOpReadVariableOp)dense_263_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_263/BiasAdd/ReadVariableOp©
dense_263/BiasAddBiasAdddense_263/MatMul:product:0(dense_263/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_263/BiasAdd
dense_263/SoftmaxSoftmaxdense_263/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_263/SoftmaxÓ
IdentityIdentitydense_263/Softmax:softmax:00^batch_normalization_65/batchnorm/ReadVariableOp2^batch_normalization_65/batchnorm/ReadVariableOp_12^batch_normalization_65/batchnorm/ReadVariableOp_24^batch_normalization_65/batchnorm/mul/ReadVariableOp!^dense_260/BiasAdd/ReadVariableOp ^dense_260/MatMul/ReadVariableOp!^dense_261/BiasAdd/ReadVariableOp ^dense_261/MatMul/ReadVariableOp!^dense_262/BiasAdd/ReadVariableOp ^dense_262/MatMul/ReadVariableOp!^dense_263/BiasAdd/ReadVariableOp ^dense_263/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ4::::::::::::2b
/batch_normalization_65/batchnorm/ReadVariableOp/batch_normalization_65/batchnorm/ReadVariableOp2f
1batch_normalization_65/batchnorm/ReadVariableOp_11batch_normalization_65/batchnorm/ReadVariableOp_12f
1batch_normalization_65/batchnorm/ReadVariableOp_21batch_normalization_65/batchnorm/ReadVariableOp_22j
3batch_normalization_65/batchnorm/mul/ReadVariableOp3batch_normalization_65/batchnorm/mul/ReadVariableOp2D
 dense_260/BiasAdd/ReadVariableOp dense_260/BiasAdd/ReadVariableOp2B
dense_260/MatMul/ReadVariableOpdense_260/MatMul/ReadVariableOp2D
 dense_261/BiasAdd/ReadVariableOp dense_261/BiasAdd/ReadVariableOp2B
dense_261/MatMul/ReadVariableOpdense_261/MatMul/ReadVariableOp2D
 dense_262/BiasAdd/ReadVariableOp dense_262/BiasAdd/ReadVariableOp2B
dense_262/MatMul/ReadVariableOpdense_262/MatMul/ReadVariableOp2D
 dense_263/BiasAdd/ReadVariableOp dense_263/BiasAdd/ReadVariableOp2B
dense_263/MatMul/ReadVariableOpdense_263/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*²
serving_default
A
input_665
serving_default_input_66:0ÿÿÿÿÿÿÿÿÿ4=
	dense_2630
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¿ã
û=
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
v__call__"Ó:
_tf_keras_network·:{"class_name": "Functional", "name": "model_65", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_65", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_66"}, "name": "input_66", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_65", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_65", "inbound_nodes": [[["input_66", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_260", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_260", "inbound_nodes": [[["flatten_65", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_261", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_261", "inbound_nodes": [[["dense_260", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_262", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_262", "inbound_nodes": [[["dense_261", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_65", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_65", "inbound_nodes": [[["dense_262", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_263", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_263", "inbound_nodes": [[["batch_normalization_65", 0, 0, {}]]]}], "input_layers": [["input_66", 0, 0]], "output_layers": [["dense_263", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 52, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_65", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_66"}, "name": "input_66", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_65", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_65", "inbound_nodes": [[["input_66", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_260", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_260", "inbound_nodes": [[["flatten_65", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_261", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_261", "inbound_nodes": [[["dense_260", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_262", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_262", "inbound_nodes": [[["dense_261", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_65", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_65", "inbound_nodes": [[["dense_262", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_263", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_263", "inbound_nodes": [[["batch_normalization_65", 0, 0, {}]]]}], "input_layers": [["input_66", 0, 0]], "output_layers": [["dense_263", 0, 0]]}}, "training_config": {"loss": "loss_func", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "clipnorm": 1, "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ó"ð
_tf_keras_input_layerÐ{"class_name": "InputLayer", "name": "input_66", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 52, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_66"}}
è
regularization_losses
	variables
trainable_variables
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"Ù
_tf_keras_layer¿{"class_name": "Flatten", "name": "flatten_65", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_65", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*y&call_and_return_all_conditional_losses
z__call__"ó
_tf_keras_layerÙ{"class_name": "Dense", "name": "dense_260", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_260", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 156}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 156]}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*{&call_and_return_all_conditional_losses
|__call__"ó
_tf_keras_layerÙ{"class_name": "Dense", "name": "dense_261", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_261", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}


kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
*}&call_and_return_all_conditional_losses
~__call__"ó
_tf_keras_layerÙ{"class_name": "Dense", "name": "dense_262", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_262", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
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
_tf_keras_layerÈ{"class_name": "BatchNormalization", "name": "batch_normalization_65", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_65", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ú

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+&call_and_return_all_conditional_losses
__call__"Ó
_tf_keras_layer¹{"class_name": "Dense", "name": "dense_263", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_263", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
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
2dense_260/kernel
:2dense_260/bias
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
2dense_261/kernel
:2dense_261/bias
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
2dense_262/kernel
:2dense_262/bias
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
+:)2batch_normalization_65/gamma
*:(2batch_normalization_65/beta
3:1 (2"batch_normalization_65/moving_mean
7:5 (2&batch_normalization_65/moving_variance
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
#:!	2dense_263/kernel
:2dense_263/bias
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
2Adam/dense_260/kernel/m
": 2Adam/dense_260/bias/m
):'
2Adam/dense_261/kernel/m
": 2Adam/dense_261/bias/m
):'
2Adam/dense_262/kernel/m
": 2Adam/dense_262/bias/m
0:.2#Adam/batch_normalization_65/gamma/m
/:-2"Adam/batch_normalization_65/beta/m
(:&	2Adam/dense_263/kernel/m
!:2Adam/dense_263/bias/m
):'
2Adam/dense_260/kernel/v
": 2Adam/dense_260/bias/v
):'
2Adam/dense_261/kernel/v
": 2Adam/dense_261/bias/v
):'
2Adam/dense_262/kernel/v
": 2Adam/dense_262/bias/v
0:.2#Adam/batch_normalization_65/gamma/v
/:-2"Adam/batch_normalization_65/beta/v
(:&	2Adam/dense_263/kernel/v
!:2Adam/dense_263/bias/v
ê2ç
G__inference_model_65_layer_call_and_return_conditional_losses_675710310
G__inference_model_65_layer_call_and_return_conditional_losses_675710542
G__inference_model_65_layer_call_and_return_conditional_losses_675710592
G__inference_model_65_layer_call_and_return_conditional_losses_675710276À
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
$__inference__wrapped_model_675709974»
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
input_66ÿÿÿÿÿÿÿÿÿ4
þ2û
,__inference_model_65_layer_call_fn_675710650
,__inference_model_65_layer_call_fn_675710621
,__inference_model_65_layer_call_fn_675710374
,__inference_model_65_layer_call_fn_675710437À
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
I__inference_flatten_65_layer_call_and_return_conditional_losses_675710656¢
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
.__inference_flatten_65_layer_call_fn_675710661¢
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
ò2ï
H__inference_dense_260_layer_call_and_return_conditional_losses_675710672¢
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
×2Ô
-__inference_dense_260_layer_call_fn_675710681¢
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
ò2ï
H__inference_dense_261_layer_call_and_return_conditional_losses_675710692¢
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
×2Ô
-__inference_dense_261_layer_call_fn_675710701¢
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
ò2ï
H__inference_dense_262_layer_call_and_return_conditional_losses_675710712¢
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
×2Ô
-__inference_dense_262_layer_call_fn_675710721¢
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
U__inference_batch_normalization_65_layer_call_and_return_conditional_losses_675710777
U__inference_batch_normalization_65_layer_call_and_return_conditional_losses_675710757´
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
:__inference_batch_normalization_65_layer_call_fn_675710803
:__inference_batch_normalization_65_layer_call_fn_675710790´
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
ò2ï
H__inference_dense_263_layer_call_and_return_conditional_losses_675710814¢
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
×2Ô
-__inference_dense_263_layer_call_fn_675710823¢
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
'__inference_signature_wrapper_675710476input_66"
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
 ¤
$__inference__wrapped_model_675709974|(%'&-.5¢2
+¢(
&#
input_66ÿÿÿÿÿÿÿÿÿ4
ª "5ª2
0
	dense_263# 
	dense_263ÿÿÿÿÿÿÿÿÿ½
U__inference_batch_normalization_65_layer_call_and_return_conditional_losses_675710757d'(%&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ½
U__inference_batch_normalization_65_layer_call_and_return_conditional_losses_675710777d(%'&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
:__inference_batch_normalization_65_layer_call_fn_675710790W'(%&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
:__inference_batch_normalization_65_layer_call_fn_675710803W(%'&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿª
H__inference_dense_260_layer_call_and_return_conditional_losses_675710672^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_dense_260_layer_call_fn_675710681Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
H__inference_dense_261_layer_call_and_return_conditional_losses_675710692^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_dense_261_layer_call_fn_675710701Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
H__inference_dense_262_layer_call_and_return_conditional_losses_675710712^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_dense_262_layer_call_fn_675710721Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
H__inference_dense_263_layer_call_and_return_conditional_losses_675710814]-.0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_dense_263_layer_call_fn_675710823P-.0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
I__inference_flatten_65_layer_call_and_return_conditional_losses_675710656]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ4
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_flatten_65_layer_call_fn_675710661P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ4
ª "ÿÿÿÿÿÿÿÿÿ¿
G__inference_model_65_layer_call_and_return_conditional_losses_675710276t'(%&-.=¢:
3¢0
&#
input_66ÿÿÿÿÿÿÿÿÿ4
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¿
G__inference_model_65_layer_call_and_return_conditional_losses_675710310t(%'&-.=¢:
3¢0
&#
input_66ÿÿÿÿÿÿÿÿÿ4
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
G__inference_model_65_layer_call_and_return_conditional_losses_675710542r'(%&-.;¢8
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
G__inference_model_65_layer_call_and_return_conditional_losses_675710592r(%'&-.;¢8
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
,__inference_model_65_layer_call_fn_675710374g'(%&-.=¢:
3¢0
&#
input_66ÿÿÿÿÿÿÿÿÿ4
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_model_65_layer_call_fn_675710437g(%'&-.=¢:
3¢0
&#
input_66ÿÿÿÿÿÿÿÿÿ4
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_model_65_layer_call_fn_675710621e'(%&-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_model_65_layer_call_fn_675710650e(%'&-.;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ4
p 

 
ª "ÿÿÿÿÿÿÿÿÿ´
'__inference_signature_wrapper_675710476(%'&-.A¢>
¢ 
7ª4
2
input_66&#
input_66ÿÿÿÿÿÿÿÿÿ4"5ª2
0
	dense_263# 
	dense_263ÿÿÿÿÿÿÿÿÿ