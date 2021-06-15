import numpy as np
import pandas as pd
import torch


class SessionDataset:
    def __init__(self, sep='\t', session_key='SessionId', item_key='ItemId', time_key='TimeStamp', user_key='UserId',
                 mode='train', train_data=None, test_data=None, n_samples=-1, itemmap=None, time_sort=False, print_info=True):
        """
        Args:
            path: path of the csv file
            sep: separator for the csv
            session_key, item_key, time_key: name of the fields corresponding to the sessions, items, time
            n_samples: the number of samples to use. If -1, use the whole dataset.
            itemmap: mapping between item IDs and item indices
            time_sort: whether to sort the sessions by time or not
        """
        self.mode = mode
        train_data.loc[:, 'in_eval'] = False
        if mode == 'train':
            self.df = train_data

        elif mode == 'test':
            test_users = test_data[user_key].unique()
            train_data = train_data[train_data[user_key].isin(test_users)].copy()
            test_data.loc[:, 'in_eval'] = True
            test_data = pd.concat([train_data, test_data], sort=False)
            self.df = test_data
        else:
            raise NotImplementedError('Unsupported dataset mode')

        self.session_key = session_key
        self.item_key = item_key
        self.user_key = user_key
        self.time_key = time_key
        self.time_sort = time_sort
        self.max_session_length= self.df.groupby(self.session_key).count()[self.item_key].max()
        # sampling
        if n_samples > 0: self.df = self.df[:n_samples] 
        # Add item indices
        self.add_item_indices(itemmap=itemmap)
        """
        Sort the df by time, and then by session ID. That is, df is sorted by session ID and
        clicks within a session are next to each other, where the clicks within a session are time-ordered.
        """
        self.create_offsets()
        if print_info:
            self.print_info()
        
          
    def create_offsets(self):
        self.df.sort_values([self.user_key, self.session_key, self.time_key], inplace=True)
        self.user_indptr, self.session_offsets = self.get_user_session_offsets()
        self.click_offsets = self.get_click_offsets()
        self.session_idx_arr = self.order_session_idx()
        self.user_idx_arr = self.order_user_idx()
    
    def select_test_users(self, user_slice, batch_size=100):
        if user_slice.shape[0] % batch_size == 0 and user_slice.shape[0] > 0:
            self.df.loc[~self.df[self.user_key].isin(user_slice),'in_eval'] = False
            
        else:
            self.df.loc[~self.df[self.user_key].isin(user_slice),'in_eval'] = False
            users_to_complete = batch_size - (user_slice.shape[0] % batch_size)
            
            users = self.df[~self.df[self.user_key].isin(user_slice)][self.user_key].unique()[:users_to_complete]
            self.df = pd.concat([self.df[self.df[self.user_key].isin(users)].copy(), 
                                 self.df[self.df['in_eval']].copy()],
                                ignore_index=True)
            
            self.create_offsets()
            
        self.print_info()
        
    def select_test_sessions(self, session_slice, batch_size=100):
        self.df.loc[~self.df[self.session_key].isin(session_slice),'in_eval'] = False
        #user_slice = self.df[self.df['in_eval']==True][self.user_key].unique()
        #self.select_test_users(user_slice, batch_size)
         
    def complete_test_sessions(self):
        # TODO: remove on final commit
        in_eval_rows = self.df[self.df['in_eval']==True].copy()
        test_session_length = in_eval_rows.groupby(self.session_key).count().reset_index()
        max_test_session_length = test_session_length[self.item_key].max()
        test_sessions_ids = in_eval_rows[self.session_key].unique()
        dummy_rows =in_eval_rows.drop_duplicates(self.session_key,keep='last').reset_index(drop=True)
        dummy_rows[self.time_key] = dummy_rows[self.time_key]+1e4
        dummy_rows['in_eval'] = False
        new_rows = pd.DataFrame([])
        for id_count, session_id in enumerate(test_sessions_ids):
            dummy_row =  dummy_rows.iloc[id_count]
            
            nrows = max_test_session_length - test_session_length.iloc[id_count][self.item_key]
            if nrows > 0 :
                rows_to_add = [dummy_row]*nrows
                new_rows = new_rows.append(rows_to_add, ignore_index=True)
                
                #print([dummy_row[self.session_key],session_id,
                #       test_session_length.iloc[id_count][self.item_key],
                #       nrows,len(rows_to_add),new_rows.shape[0]])
        a=self.df.shape[0]        
        print([a,new_rows.shape[0]])                
        self.df = self.df.append(new_rows,ignore_index=True)
                   
        self.create_offsets()
        print([self.df.shape[0],new_rows.shape[0],self.df.shape[0]-a])    
        #print(self.df[self.df[self.session_key].isin(test_sessions_ids)].groupby(self.session_key).count().describe())
        
        
    def print_info(self):
        test_rows = self.df[self.df['in_eval']==True].shape[0]
        print('Dataset: Users:{} Items:{} Sessions:{} Rows:{} Test-Rows: {}'.format(self.df[self.user_key].nunique(), 
                                                              self.df[self.item_key].nunique(),
                                                              self.df[self.session_key].nunique(), 
                                                                     self.df.shape[0],test_rows))
		
        
    def get_click_offsets(self):
        """
        Return the offsets of the beginning clicks of each session IDs,
        where the offset is calculated against the first click of the first session ID.
        """
        offsets = np.zeros(self.df[self.session_key].nunique() + 1, dtype=np.int32)
        # group & sort the df by session_key and get the offset values
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()

        return offsets

    def get_user_session_offsets(self):
        """
        Return the offsets of the beginning clicks of each user,
        where the offset is calculated against the first click of the first session ID of the user.
        """
        #Ordering dataset by user_key, session_key, time_key
        self.df.sort_values([self.user_key, self.session_key, self.time_key], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        #Creation arrays for indexes
        user_offsets = np.zeros(self.df[self.user_key].nunique()+1, dtype=np.int32)
        session_offsets = np.zeros(self.df[self.session_key].nunique()+1 , dtype=np.int32)
        # group & sort the df by user_key, session_key and get the offset values
        session_offsets[1:] = self.df.groupby([self.user_key, self.session_key], sort=False).size().cumsum()
        # group & sort the df by user_key and get the offset values
        user_offsets[1:] = self.df.groupby(self.user_key, sort=False)[self.session_key].nunique().cumsum()
        return user_offsets, session_offsets

    def order_session_idx(self):
        """ Order the session indices """
        if self.time_sort:
            # starting time for each sessions, sorted by session IDs
            sessions_start_time = self.df.groupby(self.session_key)[self.time_key].min().values
            # order the session indices by session starting times
            session_idx_arr = np.argsort(sessions_start_time)
        else:
            session_idx_arr = np.arange(self.df[self.session_key].nunique())

        return session_idx_arr

    def order_user_idx(self):
        """ Order the user indices """
        if self.time_sort:
            users_start_time = self.df.groupby(self.user_key)[self.time_key].min().values
            # order the session indices by user starting times
            user_idx_arr = np.argsort(users_start_time)
        else:
            user_idx_arr = np.arange(self.df[self.user_key].nunique())
        return user_idx_arr

    def add_item_indices(self, itemmap=None):
        """ 
        Add item index column named "item_idx" to the df
        Args:
            itemmap (pd.DataFrame): mapping between the item Ids and indices
        """
        if itemmap is None:
            item_ids = self.df[self.item_key].unique()  # unique item ids
            item2idx = pd.Series(data=np.arange(len(item_ids)),
                                 index=item_ids)
            itemmap = pd.DataFrame({self.item_key:item_ids,
                                   'item_idx':item2idx[item_ids].values})
        
        self.itemmap = itemmap
        self.df = pd.merge(self.df, self.itemmap, on=self.item_key, how='inner')
        
    
    @property    
    def items(self):
        return self.itemmap[self.item_key].unique()
 

class SessionDataLoader(torch.utils.data.IterableDataset):
    def __init__(self, dataset, batch_size=50,return_extras=False):
        """
        A class for creating session-parallel mini-batches.

        Args:
             dataset (SessionDataset): the session dataset to generate the batches from
             batch_size (int): size of the batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.return_extras = return_extras
        self.eval_data_cols = [dataset.session_key, dataset.user_key, dataset.item_key]
        self.train_data_cols = []
        # variables to manage iterations over users
        self.user_indptr = self.dataset.user_indptr

        self.n_users = len(self.dataset.user_indptr)
        self.session_offsets = self.dataset.session_offsets
        self.offset_users = self.session_offsets[self.user_indptr]
        self.user_idx_arr = np.arange(self.n_users - 1)
        self.user_iters = np.arange(self.batch_size)
        self.user_maxiter = self.user_iters.max()
        self.user_start = self.offset_users[self.user_idx_arr[self.user_iters]]
        self.user_end = self.offset_users[self.user_idx_arr[self.user_iters] + 1]

        self.session_iters = self.user_indptr[self.user_iters]
        self.session_start = self.session_offsets[self.session_iters]
        self.session_end = self.session_offsets[self.session_iters + 1]

    def __len__(self):
        return int((self.dataset.df.shape[0]/self.batch_size) +1)

    def preprocess_data(self, data):
        # sort by user and time key in order
        user_key, item_key, session_key, time_key = self.dataset.user_key, \
                                                    self.dataset.item_key, \
                                                    self.dataset.session_key, \
                                                    self.dataset.time_key

        data.sort_values([user_key, session_key, time_key], inplace=True)
        data.reset_index(drop=True, inplace=True)
        offset_session = np.r_[0, data.groupby([user_key, session_key], sort=False).size().cumsum()[:-1]]
        user_indptr = np.r_[0, data.groupby(user_key, sort=False)[session_key].nunique().cumsum()[:-1]]
        return user_indptr, offset_session
    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.

        Yields:
            input (B,): torch.FloatTensor. Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """
        mode = self.dataset.mode
        # initializations
        df = self.dataset.df
        user_indptr = self.dataset.user_indptr


        # variables to manage iterations over users
        user_indptr, session_offsets = self.preprocess_data(df)
        n_users = len(self.dataset.user_indptr)
        #session_offsets = self.dataset.session_offsets
        offset_users = session_offsets[user_indptr]
        user_idx_arr = np.arange(n_users-1)
        user_iters = np.arange(self.batch_size)
        user_maxiter = user_iters.max()
        user_start = offset_users[user_idx_arr[user_iters]]
        user_end = offset_users[user_idx_arr[user_iters] + 1]

        session_iters = user_indptr[user_iters]
        session_start = session_offsets[session_iters]
        session_end = session_offsets[session_iters + 1]

        sstart = np.zeros((self.batch_size,), dtype=np.float32)
        ustart = np.zeros((self.batch_size,), dtype=np.float32)

        #mask_zerofill = np.array([True]*self.batch_size, dtype=np.float32)
        finished = False
        while not finished:

            session_minlen = (session_end - session_start).min()
            
            idx_target = df.item_idx.values[session_start]
            for i in range(session_minlen - 1):
                # Build inputs & targets

                idx_input = idx_target
                idx_target = df.item_idx.values[session_start + i + 1]
                input_id = torch.LongTensor(idx_input)
                target = torch.LongTensor(idx_target)
                # Retrieving extra columns for various purposes
                data = self.collect_extra_columns(session_start+i+1, mode=mode)
                yield input_id, target, sstart, ustart, data
                sstart = np.zeros_like(sstart, dtype=np.float32)
                ustart = np.zeros_like(ustart, dtype=np.float32)
            
            session_start = session_start + session_minlen -1
            session_start_mask = np.arange(len(session_iters))[(session_end - session_start) <= 1]
            sstart[session_start_mask] = 1
            for idx in session_start_mask:
                session_iters[idx] += 1
                if session_iters[idx] + 1 >= len(session_offsets):
                    # retreiving previous rows keeping end the same end
                    #session_start[idx] -= session_end.max() 
                    #mask_zerofill[idx] = False
                    #if session_end.max()==df.shape[0]:
                    #   break
                    finished = True
                    break
                session_start[idx] = session_offsets[session_iters[idx]]
                session_end[idx] = session_offsets[session_iters[idx] + 1]

            # reset the User hidden state at user change
            user_change_mask = np.arange(len(user_iters))[(user_end - session_start <= 0)]
            ustart[user_change_mask] = 1
            for idx in user_change_mask:
                user_maxiter += 1
                if user_maxiter + 1 >= len(offset_users):
                    #mask_zerofill[idx] = False
                    #user_end[idx] = df.shape[0]-1
                    #session_iters[idx]
                    #session_start[idx]=0
                    #session_end[idx]=df.shape[0]
                    print(['user_end', len(offset_users), user_maxiter])
                    finished = True
                    break
                user_iters[idx] = user_maxiter
                user_start[idx] = offset_users[user_maxiter]
                user_end[idx] = offset_users[user_maxiter + 1]
                session_iters[idx] = user_indptr[user_maxiter]
                session_start[idx] = session_offsets[session_iters[idx]]
                session_end[idx] = session_offsets[session_iters[idx] + 1]

    def skip_sessions(self, minibatch_ids):
        return 

    def generate_batches(self):
        self.batches = []
        for batch in self:
            self.batches.append(batch)
        print('Batches in Dataset:{}'.format(len(self.batches)))
        
    def get_batches(self, batch_id_start, batch_id_end):
        batches = []
        if batch_id_end < len(self.batches) and batch_id_start <len(self.batches):
            batches = self.batches[batch_id_start:batch_id_end]
        else:
            batches = self.batches[batch_id_start:]
        if len(batches) == 0:
            raise AssertionError('batches collecting failed from indexes {} to {}'.
                                 format(batch_id_start, batch_id_end))
        return batches
            
    def collect_eval_data(self, row_indexes, key='in_eval'):
        eval_data = {}
        eval_mask = self.dataset.df[key].values[row_indexes]
        eval_data[key] = np.arange(self.batch_size, dtype=np.int32)[eval_mask]

        data_src = self.dataset.df.iloc[row_indexes].copy()

        for col in self.eval_data_cols:
            eval_col = data_src[col].values[eval_mask]
            eval_data[col] = eval_col

        return eval_data

    def collect_extra_columns(self,row_indexes, mode='train'):
        data = None
        if mode == 'train':
            if len(self.train_data_cols) > 0:
                rows = self.dataset.df.iloc[row_indexes].copy()
                data = {col: rows[col].values for col in self.train_data_cols}
        elif mode == 'test':
            data = self.collect_eval_data(row_indexes)
        else:
            raise NotImplementedError('Not implemented dataset mode')
        return data
